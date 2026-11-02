"""大肥鱼天气 · AstrBot 插件入口。

功能：
1. 自然语言识别天气问题（不需要固定指令），主人/白名单 QQ 才生效
2. 第一次查询时用人设口吻问一次城市，之后自动绑定
3. 中国天气网 / Open-Meteo / wttr.in 多源自动切换
4. 用当前人格润色天气播报，失败时回落到模板
5. 附带 get_weather LLM 工具、/天气 指令和可视化面板
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime
from typing import Any, AsyncGenerator

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import Location as LocationComponent
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.web import error_response, json_response, request

try:  # 以包方式加载（正常情况）
    from . import weather_core as core
except ImportError:  # 直接运行测试脚本时
    import weather_core as core  # type: ignore

PLUGIN_NAME = "astrbot_plugin_dafeiyu_weather"
PLUGIN_DISPLAY_NAME = "大肥鱼天气"

# 允许在群里不 @ 也响应？默认关闭，避免刷屏
_REBIND_RE = re.compile(
    r"(?:以后|今后|往后)(?:都|就)?(?:查|看|用)\s*([\u4e00-\u9fa5A-Za-z·]{2,10})"
    r"|(?:改成|改到|换到|换去|绑定到|绑定|记住|记一下|记下|搬到|搬去|我在|我住在|住在)"
    r"\s*([\u4e00-\u9fa5A-Za-z·]{2,10})"
)

_COMMAND_WORDS = ("天气绑定", "天气解绑", "天气")
_ALIAS_QUERY_WORDS = {"老家", "公司", "学校", "单位", "家", "爸妈家", "娘家", "婆家", "新家"}


def _looks_like_command(text: str) -> bool:
    """这条消息会不会被指令处理器接走。"""
    stripped = (text or "").strip()
    for word in _COMMAND_WORDS:
        if stripped == word:
            return True
        if (
            stripped.startswith(word)
            and len(stripped) > len(word)
            and stripped[len(word)] in " \t:："
        ):
            return True
    return False


def _fmt_timestamp(value: Any) -> str:
    try:
        return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


class DafeiyuWeather(Star):
    """大肥鱼天气：自然语言查天气，第一次问一次城市，之后自动记住。

    直接说话就行，比如「今天天气怎么样」「明天会下雨吗」「杭州天气」。
    也支持 /天气、/天气绑定 杭州、/天气解绑。
    """

    def __init__(self, context: Context, config: Any = None) -> None:
        super().__init__(context, config)
        self.config = config
        try:
            data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        except Exception:  # noqa: BLE001 - 老版本或异常时退回插件目录
            data_dir = os.path.join(os.path.dirname(__file__), "data")
            os.makedirs(data_dir, exist_ok=True)

        self.store = core.BindingStore(os.path.join(str(data_dir), "bindings.json"))
        self.data_dir = str(data_dir)
        self.service = core.WeatherService(self._cfg)
        # (umo, sender_id) -> {"expire": ts, "kind": "location"/"confirm_city"}
        self._pending: dict[tuple[str, str], dict[str, Any]] = {}
        # (umo, sender_id) -> 上一次查询上下文，用于「那明天呢」这类追问
        self._last_query: dict[tuple[str, str], dict[str, Any]] = {}
        self._reminder_job: Any = None
        self._register_web_api()

    async def initialize(self) -> None:
        await self._setup_reminder_job()

    async def terminate(self) -> None:
        await self._remove_reminder_job()

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------
    def _cfg(self, path: str, default: Any = None) -> Any:
        if self.config is None:
            return default
        value: Any = self.config
        for part in path.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(part)
            if value is None:
                return default
        return default if value is None else value

    def _cfg_bool(self, path: str, default: bool) -> bool:
        value = self._cfg(path, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on", "是")
        return bool(value)

    def _cfg_int(self, path: str, default: int) -> int:
        try:
            return int(self._cfg(path, default))
        except (TypeError, ValueError):
            return default

    @property
    def _allowed_users(self) -> set[str]:
        raw = self._cfg("basic.allowed_users", [])
        if isinstance(raw, str):
            raw = re.split(r"[\s,，;；]+", raw)
        if not isinstance(raw, (list, tuple, set)):
            return set()
        result: set[str] = set()
        for item in raw:
            text = str(item).strip()
            if text:
                result.add(text)
        return result

    # ------------------------------------------------------------------
    # 事件入口
    # ------------------------------------------------------------------
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """自然语言天气入口。"""
        try:
            async for item in self._dispatch(event):
                yield item
        except Exception:  # noqa: BLE001 - 插件错误不能拖垮整条消息链
            logger.exception("[%s] 处理消息失败", PLUGIN_DISPLAY_NAME)

    async def _dispatch(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        text = (event.message_str or "").strip()
        card = self._location_card(event)
        if not text and card is None:
            return
        if not self._is_allowed(event):
            return
        if self._need_at(event):
            return

        sender = str(event.get_sender_id() or "")
        if not sender:
            return
        key = (event.unified_msg_origin, sender)
        now = time.time()

        # 0) QQ 位置卡片
        if card is not None and self._cfg_bool("features.location_card", True):
            async for item in self._handle_location_card(event, sender, card):
                yield item
            return

        if text.startswith(("/", "／")):
            return
        if self._cfg_bool("features.enable_commands", True) and _looks_like_command(
            text
        ):
            # 交给指令处理器，避免同一条消息回两遍
            return

        pending = self._pending.get(key)
        if pending:
            if pending.get("expire", 0) <= now:
                self._pending.pop(key, None)
            else:
                async for item in self._handle_pending(event, text, key, sender):
                    yield item
                return

        binding = await self.store.get(sender)

        # 1) 别名绑定：「记住老家是临沂」
        alias_hit = core.extract_alias_binding(text)
        if alias_hit and self._cfg_bool("features.aliases", True):
            alias, city = alias_hit
            location = await self.service.resolve(city)
            if location is not None:
                await self._set_alias(sender, alias, location, event)
                reply = await self._confirm_alias_text(event, alias, location)
                event.stop_event()
                yield self._result(event, reply)
                return

        # 2) 追问：「那明天呢」「会下吗」
        follow = core.detect_followup(text)
        if follow and self._cfg_bool("features.followup", True):
            last = self._last_query.get(key)
            if last and now - float(last.get("ts") or 0) <= max(
                1, self._cfg_int("features.followup_minutes", 10)
            ) * 60:
                location = core.Location.from_dict(last.get("location") or {})
                scope = follow.get("scope") or last.get("scope") or "today"
                aspects = set(follow.get("aspects") or [])
                async for item in self._answer_weather(
                    event, key, sender, location, text, scope, aspects
                ):
                    yield item
                return

        # 3) 出门场景：「我出门了」「去上班了」
        if core.is_departure(text) and self._cfg_bool(
            "features.departure_scene", True
        ):
            location = await self._default_location(binding)
            if location is None:
                question = await self._ask_location_pending(event, key)
                if question:
                    event.stop_event()
                    yield event.plain_result(question)
                return
            async for item in self._answer_departure(
                event, key, sender, location, text
            ):
                yield item
            return

        # 4) 小时级降水：「几点下雨」「雨什么时候停」
        if core.detect_hourly_intent(text) and self._cfg_bool(
            "features.hourly_rain", True
        ):
            location = await self._resolve_from_text(
                text, sender
            ) or await self._default_location(binding)
            if location is None:
                question = await self._ask_location_pending(event, key)
                if question:
                    event.stop_event()
                    yield event.plain_result(question)
                return
            async for item in self._answer_hourly(
                event, key, sender, location, text
            ):
                yield item
            return

        # 5) 「以后查杭州」「改到杭州」这类改绑，不需要带“天气”两个字
        rebind_city = self._extract_rebind_city(text)
        explicit_rebind = bool(
            re.search(
                r"(?:以后|今后|往后).{0,8}(?:查|看|用)|改成|改到|换到|绑定|记住|搬到",
                text,
            )
        )
        implicit_rebind = bool(binding and re.search(r"我在|住在", text))
        if rebind_city and (explicit_rebind or implicit_rebind):
            found = await self.service.resolve(rebind_city)
            if found is not None:
                await self._bind(sender, event, found)
                if core.detect_weather_intent(text):
                    async for item in self._answer_weather(
                        event,
                        key,
                        sender,
                        found,
                        text,
                        core.detect_time_scope(text),
                        core.detect_aspects(text),
                        first_time=True,
                    ):
                        yield item
                else:
                    reply = await self._confirm_bind_text(event, found)
                    event.stop_event()
                    yield self._result(event, reply)
                return

        if not core.detect_weather_intent(text):
            return

        scope = core.detect_time_scope(text)
        aspects = core.detect_aspects(text)
        location = await self._resolve_from_text(text, sender)
        explicit = location is not None
        candidates = core.extract_city_candidates(text)

        # 猜错别字/同音字：杭洲 -> 杭州（有绑定也先猜，避免把“杭洲”当成本地）
        if (
            location is None
            and candidates
            and candidates[0] not in _ALIAS_QUERY_WORDS
            and self._cfg_bool("features.typo_guess", True)
        ):
            guess = await self._guess_city(event, candidates[0])
            if guess is not None:
                question = await self._confirm_city_text(event, guess.name)
                self._pending[key] = {
                    "expire": now
                    + max(1, self._cfg_int("basic.ask_timeout", 10)) * 60,
                    "kind": "confirm_city",
                    "location": guess.to_dict(),
                }
                event.stop_event()
                yield event.plain_result(question)
                return

        if location is None and binding:
            location = core.Location.from_dict(binding.get("location") or {})
        if location is None:
            location = await self._default_location(None)

        if location is None:
            question = await self._ask_location_pending(event, key)
            if question is None:
                event.stop_event()
                yield event.plain_result("我这儿没记着你在哪个城市，先说个城市名吧")
            else:
                event.stop_event()
                yield event.plain_result(question)
            return

        first_time = binding is None
        if explicit and binding is None:
            await self._bind(sender, event, location)
            first_time = True
        async for item in self._answer_weather(
            event,
            key,
            sender,
            location,
            text,
            scope,
            aspects,
            first_time=first_time,
        ):
            yield item

    async def _handle_pending(
        self,
        event: AstrMessageEvent,
        text: str,
        key: tuple[str, str],
        sender: str,
    ) -> AsyncGenerator[Any, None]:
        pending = self._pending.get(key) or {}
        kind = pending.get("kind") or "location"

        if core.is_cancel(text):
            self._pending.pop(key, None)
            event.stop_event()
            yield event.plain_result("好，那先不查啦")
            return

        # 「你是说杭州吗」的确认
        if kind == "confirm_city":
            if core.is_confirm_yes(text):
                location = core.Location.from_dict(pending.get("location") or {})
                self._pending.pop(key, None)
                await self._bind(sender, event, location)
                async for item in self._answer_weather(
                    event,
                    key,
                    sender,
                    location,
                    text,
                    core.detect_time_scope(text),
                    core.detect_aspects(text),
                    first_time=True,
                ):
                    yield item
                return
            if core.is_confirm_no(text):
                self._pending.pop(key, None)
                question = await self._ask_location_pending(event, key)
                if question:
                    event.stop_event()
                    yield event.plain_result(question)
                return
            self._pending.pop(key, None)

        # 又问了一遍天气（可能这次带了城市）
        if core.detect_weather_intent(text):
            location = await self._resolve_from_text(text, sender)
            if location is not None:
                self._pending.pop(key, None)
                await self._bind(sender, event, location)
                async for item in self._answer_weather(
                    event,
                    key,
                    sender,
                    location,
                    text,
                    core.detect_time_scope(text),
                    core.detect_aspects(text),
                    first_time=True,
                ):
                    yield item
                return
            question = await self._ask_location_pending(event, key)
            if question:
                event.stop_event()
                yield event.plain_result(question)
            return

        cleaned = core.clean_location_answer(text)
        if not core.looks_like_location(cleaned):
            # 不是地名，也不像在回答，放回给正常聊天
            self._pending.pop(key, None)
            return

        location = await self.service.resolve(cleaned)
        if location is None:
            guess = await self._guess_city(event, cleaned)
            if guess is not None:
                question = await self._confirm_city_text(event, guess.name)
                self._pending[key] = {
                    "expire": time.time()
                    + max(1, self._cfg_int("basic.ask_timeout", 10)) * 60,
                    "kind": "confirm_city",
                    "location": guess.to_dict(),
                }
                event.stop_event()
                yield event.plain_result(question)
                return
            self._pending[key] = {
                "expire": time.time()
                + max(1, self._cfg_int("basic.ask_timeout", 10)) * 60,
                "kind": "location",
            }
            event.stop_event()
            yield event.plain_result(
                f"没找到「{cleaned}」这个地方呀\n说个城市名试试，比如杭州、临沂"
            )
            return

        await self._bind(sender, event, location)
        self._pending.pop(key, None)
        async for item in self._answer_weather(
            event,
            key,
            sender,
            location,
            text,
            core.detect_time_scope(text),
            core.detect_aspects(text),
            first_time=True,
        ):
            yield item

    # ------------------------------------------------------------------
    # 权限 / 绑定
    # ------------------------------------------------------------------
    def _is_allowed(self, event: AstrMessageEvent) -> bool:
        sender = str(event.get_sender_id() or "")
        if not sender:
            return False
        if sender in self._allowed_users:
            return True
        try:
            if event.is_admin():
                return True
        except Exception:  # noqa: BLE001
            pass
        if self._cfg_bool("basic.allow_private_all", True):
            try:
                return bool(event.is_private_chat())
            except Exception:  # noqa: BLE001
                return False
        return False

    def _need_at(self, event: AstrMessageEvent) -> bool:
        try:
            private = bool(event.is_private_chat())
        except Exception:  # noqa: BLE001
            private = True
        if private:
            return False
        if not self._cfg_bool("basic.group_require_at", True):
            return False
        try:
            return not bool(event.is_at_or_wake_command)
        except Exception:  # noqa: BLE001
            return True

    async def _bind(
        self, sender: str, event: AstrMessageEvent, location: core.Location
    ) -> None:
        record = await self.store.get(sender) or {}
        record.update(
            {
                "location": location.to_dict(),
                "updated": time.time(),
                "nickname": event.get_sender_name() or record.get("nickname", ""),
                "platform": event.get_platform_id() or record.get("platform", ""),
                "umo": event.unified_msg_origin,
            }
        )
        await self.store.set(sender, record)
        logger.info(
            "[%s] 绑定地区 %s -> %s",
            PLUGIN_DISPLAY_NAME,
            sender,
            location.display,
        )

    def _extract_rebind_city(self, text: str) -> str:
        match = _REBIND_RE.search(text or "")
        if not match:
            return ""
        name = core.strip_time_prefix(match.group(1) or match.group(2) or "")
        if not name:
            return ""
        if re.search(r"天气|气温|温度|下雨|下雪|带伞|几度|多少度|冷|热|穿", name):
            return ""
        if name in core.LOCATION_STOPWORDS:
            return ""
        return name

    # ------------------------------------------------------------------
    # 人格润色
    # ------------------------------------------------------------------
    async def _persona_prompt_for(self, umo: str) -> str:
        if not self._cfg_bool("reply.use_persona", True):
            return ""
        try:
            manager = getattr(self.context, "persona_manager", None)
            if manager is None:
                return ""
            persona = await manager.get_default_persona_v3(umo)
            if isinstance(persona, dict):
                return str(persona.get("prompt") or "").strip()
            return str(getattr(persona, "system_prompt", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s] 读取人格失败：%s", PLUGIN_DISPLAY_NAME, exc)
            return ""

    async def _llm_for(self, umo: str, system_prompt: str, prompt: str) -> str:
        try:
            provider = await self.context.get_using_provider_async(umo)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 获取对话模型失败：%s", PLUGIN_DISPLAY_NAME, exc)
            return ""
        if provider is None:
            return ""
        try:
            response = await asyncio.wait_for(
                provider.text_chat(
                    prompt=prompt,
                    system_prompt=system_prompt or None,
                ),
                timeout=max(5, self._cfg_int("reply.llm_timeout", 25)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 人格润色失败：%s", PLUGIN_DISPLAY_NAME, exc)
            return ""
        return str(getattr(response, "completion_text", "") or "").strip()

    async def _persona_prompt(self, event: AstrMessageEvent) -> str:
        return await self._persona_prompt_for(event.unified_msg_origin)

    async def _llm(
        self, event: AstrMessageEvent, system_prompt: str, prompt: str
    ) -> str:
        return await self._llm_for(event.unified_msg_origin, system_prompt, prompt)

    async def _ask_location_text(self, event: AstrMessageEvent) -> str:
        fallback = core.ask_location_template()
        if (
            str(self._cfg("reply.mode", "llm")) != "llm"
            or not self._cfg_bool("features.ask_polish", True)
        ):
            return fallback
        persona = await self._persona_prompt(event)
        prompt = core.build_ask_prompt(
            event.get_sender_name() or "",
            self._cfg_int("reply.max_chars", 200),
        )
        reply = core.clean_llm_reply(
            await self._llm(event, persona, prompt),
            self._cfg_int("reply.max_chars", 200),
        )
        return reply or fallback

    async def _confirm_bind_text(
        self, event: AstrMessageEvent, location: core.Location
    ) -> str:
        fallback = f"好，记住啦，以后都给你查{location.name}的天气"
        if str(self._cfg("reply.mode", "llm")) != "llm":
            return fallback
        persona = await self._persona_prompt(event)
        prompt = core.build_bind_prompt(
            location.name,
            event.get_sender_name() or "",
            self._cfg_int("reply.max_chars", 200),
        )
        reply = core.clean_llm_reply(
            await self._llm(event, persona, prompt),
            self._cfg_int("reply.max_chars", 200),
        )
        return reply or fallback

    # ------------------------------------------------------------------
    # 追问 / 快照 / 别名 / 位置 / 表情
    # ------------------------------------------------------------------
    def _remember_query(
        self,
        key: tuple[str, str],
        location: core.Location,
        scope: str,
        aspects: set[str] | None,
    ) -> None:
        self._last_query[key] = {
            "location": location.to_dict(),
            "scope": scope,
            "aspects": sorted(aspects or set()),
            "ts": time.time(),
        }
        if len(self._last_query) > 300:
            oldest = sorted(
                self._last_query,
                key=lambda item: self._last_query[item].get("ts", 0),
            )[:50]
            for item in oldest:
                self._last_query.pop(item, None)

    async def _remember_snapshot(
        self, sender: str, data: core.WeatherData
    ) -> None:
        record = await self.store.get(sender)
        if not record:
            return
        record["snapshot"] = {
            "city": data.location.name,
            "temp": data.current_temp,
            "high": data.today_max,
            "low": data.today_min,
            "text": data.current_text,
            "ts": time.time(),
        }
        await self.store.set(sender, record)

    async def _get_alias(
        self, sender: str, name: str
    ) -> core.Location | None:
        record = await self.store.get(sender) or {}
        aliases = record.get("aliases") or {}
        item = aliases.get(name)
        return core.Location.from_dict(item) if item else None

    async def _set_alias(
        self,
        sender: str,
        alias: str,
        location: core.Location,
        event: AstrMessageEvent,
    ) -> None:
        record = await self.store.get(sender) or {}
        aliases = record.get("aliases") or {}
        aliases[alias] = location.to_dict()
        record["aliases"] = aliases
        if not record.get("location"):
            record["location"] = location.to_dict()
        record["updated"] = time.time()
        record.setdefault("nickname", event.get_sender_name() or "")
        record.setdefault("platform", event.get_platform_id() or "")
        record.setdefault("umo", event.unified_msg_origin)
        await self.store.set(sender, record)

    async def _resolve_from_text(
        self, text: str, sender: str
    ) -> core.Location | None:
        cleaned = core.strip_weather_suffix(text)
        if core.looks_like_location(cleaned):
            alias = await self._get_alias(sender, cleaned)
            if alias is not None:
                return alias
            found = await self.service.resolve(cleaned)
            if found is not None:
                return found
        for candidate in core.extract_city_candidates(text):
            alias = await self._get_alias(sender, candidate)
            if alias is not None:
                return alias
            found = await self.service.resolve(candidate)
            if found is not None:
                return found
        return None

    async def _default_location(
        self, binding: dict[str, Any] | None
    ) -> core.Location | None:
        if binding:
            return core.Location.from_dict(binding.get("location") or {})
        default_city = str(self._cfg("basic.default_city", "") or "").strip()
        if default_city:
            return await self.service.resolve(default_city)
        return None

    async def _ask_location_pending(
        self, event: AstrMessageEvent, key: tuple[str, str]
    ) -> str | None:
        if not self._cfg_bool("basic.ask_location", True):
            return None
        question = await self._ask_location_text(event)
        self._pending[key] = {
            "expire": time.time()
            + max(1, self._cfg_int("basic.ask_timeout", 10)) * 60,
            "kind": "location",
        }
        return question

    async def _festival_note(
        self, sender: str, data: core.WeatherData
    ) -> str:
        if not self._cfg_bool("features.festival_egg", False):
            return ""
        notes: list[str] = []
        term = core.solar_term_today()
        if term:
            notes.append(f"今天是{term}，可以顺口提一句")
        texts = " ".join(
            [
                data.current_text or "",
                (data.today.text if data.today else "") or "",
            ]
        )
        if "雪" in texts:
            record = await self.store.get(sender)
            if record is not None:
                now = datetime.now()
                season = (
                    f"{now.year}-winter"
                    if now.month >= 10
                    else f"{now.year - 1}-winter"
                )
                if record.get("snow_seen") != season:
                    record["snow_seen"] = season
                    await self.store.set(sender, record)
                    notes.append("这是今年第一场雪，可以提一句")
        return "；".join(notes)

    async def _answer_weather(
        self,
        event: AstrMessageEvent,
        key: tuple[str, str],
        sender: str,
        location: core.Location,
        text: str,
        scope: str,
        aspects: set[str] | None,
        *,
        first_time: bool = False,
    ) -> AsyncGenerator[Any, None]:
        binding = await self.store.get(sender)
        snapshot = (binding or {}).get("snapshot")
        try:
            data = await self.service.get(location)
        except core.WeatherError as exc:
            logger.warning("[%s] 天气查询失败：%s", PLUGIN_DISPLAY_NAME, exc)
            reply = await self._failure_text(event, str(exc))
            event.stop_event()
            yield event.plain_result(reply)
            return
        festival_note = await self._festival_note(sender, data)
        reply = await self._compose_reply(
            event,
            data,
            text,
            scope=scope,
            aspects=aspects,
            first_time=first_time,
            snapshot=snapshot,
            festival_note=festival_note,
        )
        self._remember_query(key, location, scope, aspects)
        await self._remember_snapshot(sender, data)
        self._pending.pop(key, None)
        event.stop_event()
        yield self._result(event, reply, data)

    async def _answer_departure(
        self,
        event: AstrMessageEvent,
        key: tuple[str, str],
        sender: str,
        location: core.Location,
        text: str,
    ) -> AsyncGenerator[Any, None]:
        try:
            data = await self.service.get(location)
        except core.WeatherError as exc:
            reply = await self._failure_text(event, str(exc))
            event.stop_event()
            yield event.plain_result(reply)
            return
        reply = await self._compose_departure(event, data, text)
        self._remember_query(key, location, "now", set())
        await self._remember_snapshot(sender, data)
        event.stop_event()
        yield self._result(event, reply, data)

    async def _answer_hourly(
        self,
        event: AstrMessageEvent,
        key: tuple[str, str],
        sender: str,
        location: core.Location,
        text: str,
    ) -> AsyncGenerator[Any, None]:
        try:
            data = await self.service.get(location)
            _source, points = await self.service.hourly(location, hours=12)
        except core.WeatherError as exc:
            logger.warning("[%s] 小时级预报失败：%s", PLUGIN_DISPLAY_NAME, exc)
            async for item in self._answer_weather(
                event, key, sender, location, text, "now", {"rain"}
            ):
                yield item
            return
        reply = await self._compose_hourly(event, data, points, text)
        self._remember_query(key, location, "now", {"rain"})
        await self._remember_snapshot(sender, data)
        event.stop_event()
        yield self._result(event, reply, data)

    async def _handle_location_card(
        self,
        event: AstrMessageEvent,
        sender: str,
        card: Any,
    ) -> AsyncGenerator[Any, None]:
        key = (event.unified_msg_origin, sender)
        lat = getattr(card, "lat", None)
        lon = getattr(card, "lon", None)
        location = None
        if lat is not None and lon is not None:
            try:
                location = await core.reverse_geocode(
                    float(lat), float(lon), timeout=6, retries=0
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] 位置反查失败：%s", PLUGIN_DISPLAY_NAME, exc)
        if location is None:
            title = str(getattr(card, "title", "") or "")
            if title:
                location = await self.service.resolve(title)
        if location is None:
            reply = await self._failure_text(event, "位置解析失败")
            event.stop_event()
            yield event.plain_result(reply)
            return
        if not location.cn_code:
            normalized = await self.service.resolve(location.name)
            if normalized is not None and normalized.cn_code:
                location = normalized
        await self._bind(sender, event, location)
        async for item in self._answer_weather(
            event,
            key,
            sender,
            location,
            "我发了个位置",
            "today",
            set(),
            first_time=True,
        ):
            yield item

    def _location_card(self, event: AstrMessageEvent) -> Any:
        try:
            messages = event.get_messages()
        except Exception:  # noqa: BLE001
            return None
        for component in messages:
            if isinstance(component, LocationComponent):
                return component
        return None

    def _meme_path(self, kind: str) -> str:
        if not kind:
            return ""
        base = str(self._cfg("features.meme_dir", "") or "").strip()
        if not base:
            base = os.path.join(self.data_dir, "memes")
        for name in (kind, "default"):
            for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                path = os.path.join(base, f"{name}{ext}")
                if os.path.isfile(path):
                    return path
        return ""

    def _result(self, event: AstrMessageEvent, text: str, data: Any = None) -> Any:
        if data is not None and self._cfg_bool("features.meme", False):
            path = self._meme_path(core.pick_meme_kind(data))
            if path:
                try:
                    return event.make_result().message(text).file_image(path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[%s] 表情附图失败：%s", PLUGIN_DISPLAY_NAME, exc)
        return event.plain_result(text)

    async def _failure_text(self, event: AstrMessageEvent, error: str = "") -> str:
        fallback = core.failure_template()
        if (
            str(self._cfg("reply.mode", "llm")) != "llm"
            or not self._cfg_bool("features.personify_errors", True)
        ):
            return fallback
        persona = await self._persona_prompt(event)
        prompt = core.build_failure_prompt(error, event.get_sender_name() or "")
        reply = core.clean_llm_reply(
            await self._llm(event, persona, prompt),
            self._cfg_int("reply.max_chars", 200),
        )
        return reply or fallback

    async def _guess_city(
        self, event: AstrMessageEvent, raw: str
    ) -> core.Location | None:
        if (
            str(self._cfg("reply.mode", "llm")) != "llm"
            or not self._cfg_bool("features.typo_guess", True)
        ):
            return None
        guess = core.clean_llm_reply(
            await self._llm(event, "", core.build_city_guess_prompt(raw)), 20
        ).strip("「」“”\"' \n")
        guess = re.sub(r"[。.!！?？、，,]+$", "", guess).strip()
        if not guess or guess == "无":
            return None
        return await self.service.resolve(guess)

    async def _confirm_city_text(
        self, event: AstrMessageEvent, city: str
    ) -> str:
        fallback = f"你是说「{city}」吗"
        if str(self._cfg("reply.mode", "llm")) != "llm":
            return fallback
        persona = await self._persona_prompt(event)
        prompt = core.build_confirm_city_prompt(
            city, event.get_sender_name() or ""
        )
        reply = core.clean_llm_reply(
            await self._llm(event, persona, prompt),
            self._cfg_int("reply.max_chars", 200),
        )
        return reply or fallback

    async def _confirm_alias_text(
        self, event: AstrMessageEvent, alias: str, location: core.Location
    ) -> str:
        fallback = f"好，记住啦，{alias}就是{location.name}"
        if str(self._cfg("reply.mode", "llm")) != "llm":
            return fallback
        persona = await self._persona_prompt(event)
        prompt = (
            f"【场景】对方让你记住「{alias}」就是「{location.name}」。\n"
            "【要求】用你平时的口吻应一声，1行不超过30字，不用句号，只输出你要说的话。"
        )
        reply = core.clean_llm_reply(
            await self._llm(event, persona, prompt),
            self._cfg_int("reply.max_chars", 200),
        )
        return reply or fallback

    async def _compose_departure(
        self, event: AstrMessageEvent, data: core.WeatherData, text: str
    ) -> str:
        if str(self._cfg("reply.mode", "llm")) != "llm":
            return core.render_template(
                data,
                "now",
                {"temp"},
                show_air=self._cfg_bool("features.show_air_quality", True),
            )
        persona = await self._persona_prompt(event)
        prompt = core.build_departure_prompt(
            data,
            text,
            nickname=event.get_sender_name() or "",
            max_chars=max(60, self._cfg_int("reply.max_chars", 200)),
            extra=str(self._cfg("reply.extra_prompt", "") or ""),
        )
        reply = core.clean_llm_reply(
            await self._llm(event, persona, prompt),
            self._cfg_int("reply.max_chars", 200),
        )
        if reply:
            return reply
        return core.render_template(
            data,
            "now",
            {"temp"},
            show_air=self._cfg_bool("features.show_air_quality", True),
        )

    async def _compose_hourly(
        self,
        event: AstrMessageEvent,
        data: core.WeatherData,
        points: list[core.HourlyPoint],
        text: str,
    ) -> str:
        summary = core.summarize_hourly(points)
        fallback = (
            f"{data.location.name}{summary}" if summary else "这个我暂时查不到"
        )
        if str(self._cfg("reply.mode", "llm")) != "llm":
            return fallback
        persona = await self._persona_prompt(event)
        prompt = core.build_hourly_prompt(
            data,
            points,
            text,
            max_chars=max(60, self._cfg_int("reply.max_chars", 200)),
            extra=str(self._cfg("reply.extra_prompt", "") or ""),
        )
        reply = core.clean_llm_reply(
            await self._llm(event, persona, prompt),
            self._cfg_int("reply.max_chars", 200),
        )
        return reply or fallback

    async def _compose_reply(
        self,
        event: AstrMessageEvent,
        data: core.WeatherData,
        user_text: str,
        *,
        scope: str = "today",
        aspects: set[str] | None = None,
        first_time: bool = False,
        snapshot: dict[str, Any] | None = None,
        festival_note: str = "",
    ) -> str:
        max_chars = max(60, self._cfg_int("reply.max_chars", 200))
        show_air = self._cfg_bool("features.show_air_quality", True)
        snapshot_note = (
            core.compare_snapshot(data, snapshot)
            if self._cfg_bool("features.compare_last", True)
            else ""
        )
        extra = str(self._cfg("reply.extra_prompt", "") or "")
        if festival_note:
            extra = (extra + "；" if extra else "") + festival_note
        if str(self._cfg("reply.mode", "llm")) != "llm":
            text = core.render_template(
                data, scope=scope, aspects=aspects, show_air=show_air
            )
            if snapshot_note:
                text += f"\n{snapshot_note}"
            if festival_note:
                for part in festival_note.replace("，可以顺口提一句", "").replace(
                    "，可以提一句", ""
                ).split("；"):
                    if part.strip():
                        text += f"\n{part.strip()}"
            return text

        persona = await self._persona_prompt(event)
        if not persona:
            persona = "你是一个说话自然、简短、像在 QQ 上聊天的伙伴。"
        mood = (
            core.mood_hint(data)
            if self._cfg_bool("features.weather_mood", True)
            else ""
        )
        prompt = core.build_reply_prompt(
            data,
            user_text,
            scope=scope,
            aspects=aspects,
            nickname=event.get_sender_name() or "",
            max_chars=max_chars,
            first_time=first_time,
            extra=extra,
            show_alerts=self._cfg_bool("features.show_alerts", True),
            show_air=self._cfg_bool("features.show_air_quality", True),
            show_indices=self._cfg_bool("features.life_indices", True),
            mood=mood,
            snapshot_note=snapshot_note,
        )
        reply = core.clean_llm_reply(
            await self._llm(event, persona, prompt), max_chars
        )
        if reply:
            return reply
        text = core.render_template(
            data, scope=scope, aspects=aspects, show_air=show_air
        )
        if snapshot_note:
            text += f"\n{snapshot_note}"
        if festival_note:
            for part in festival_note.replace("，可以顺口提一句", "").replace(
                "，可以提一句", ""
            ).split("；"):
                if part.strip():
                    text += f"\n{part.strip()}"
        return text

    # ------------------------------------------------------------------
    # 指令
    # ------------------------------------------------------------------
    @filter.command("天气")
    async def weather_command(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """查天气：/天气 [城市]"""
        if not self._cfg_bool("features.enable_commands", True):
            return
        if not self._is_allowed(event):
            return
        arg = self._command_arg(event, ("天气",))
        sender = str(event.get_sender_id() or "")
        key = (event.unified_msg_origin, sender)

        location = await self._resolve_from_text(arg, sender) if arg else None
        binding = await self.store.get(sender)
        if location is None and binding:
            location = core.Location.from_dict(binding.get("location") or {})
        if location is None:
            location = await self._default_location(None)

        if location is None:
            question = await self._ask_location_pending(event, key)
            event.stop_event()
            yield event.plain_result(
                question or "我这儿没记着你在哪个城市，先说个城市名吧"
            )
            return

        first_time = bool(binding is None and arg)
        if first_time:
            await self._bind(sender, event, location)
        async for item in self._answer_weather(
            event,
            key,
            sender,
            location,
            event.message_str,
            core.detect_time_scope(event.message_str),
            core.detect_aspects(event.message_str),
            first_time=first_time,
        ):
            yield item

    @filter.command("天气绑定")
    async def bind_command(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """绑定地区：/天气绑定 杭州"""
        if not self._cfg_bool("features.enable_commands", True):
            return
        if not self._is_allowed(event):
            return
        arg = self._command_arg(event, ("天气绑定",))
        sender = str(event.get_sender_id() or "")
        key = (event.unified_msg_origin, sender)
        if not arg:
            question = await self._ask_location_pending(event, key)
            event.stop_event()
            yield event.plain_result(question or "说个城市名吧")
            return
        location = await self._resolve_from_text(arg, sender)
        if location is None:
            guess = await self._guess_city(event, arg)
            if guess is not None:
                question = await self._confirm_city_text(event, guess.name)
                self._pending[key] = {
                    "expire": time.time()
                    + max(1, self._cfg_int("basic.ask_timeout", 10)) * 60,
                    "kind": "confirm_city",
                    "location": guess.to_dict(),
                }
                event.stop_event()
                yield event.plain_result(question)
                return
            reply = await self._failure_text(event, "城市解析失败")
            event.stop_event()
            yield event.plain_result(f"没找到「{arg}」这个地方，换个城市名试试\n{reply}")
            return
        await self._bind(sender, event, location)
        reply = await self._confirm_bind_text(event, location)
        event.stop_event()
        yield self._result(event, reply)

    @filter.command("天气解绑")
    async def unbind_command(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """解除地区绑定：/天气解绑"""
        if not self._cfg_bool("features.enable_commands", True):
            return
        if not self._is_allowed(event):
            return
        sender = str(event.get_sender_id() or "")
        existed = await self.store.delete(sender)
        event.stop_event()
        yield event.plain_result(
            "好啦，不记着了，下次问天气再告诉我城市"
            if existed
            else "我这儿本来也没记着你的城市呀"
        )

    def _command_arg(self, event: AstrMessageEvent, names: tuple[str, ...]) -> str:
        text = (event.message_str or "").strip()
        for name in names:
            text = re.sub(
                r"^[/／]?" + re.escape(name) + r"[\s:：]*", "", text
            )
        return text.strip()

    # ------------------------------------------------------------------
    # 主动提醒（用 AstrBot 自带 cron，插件只负责生成内容）
    # ------------------------------------------------------------------
    async def _setup_reminder_job(self) -> None:
        if not self._cfg_bool("reminder.enabled", False):
            return
        cron = getattr(self.context, "cron_manager", None)
        add_job = getattr(cron, "add_basic_job", None)
        if add_job is None:
            logger.info(
                "[%s] 当前 AstrBot 没有可用的定时任务管理器，跳过主动提醒",
                PLUGIN_DISPLAY_NAME,
            )
            return
        expression = str(self._cfg("reminder.cron", "0 8 * * *") or "").strip()
        if not expression:
            return
        # 上次异常退出可能留下同名任务，先清掉再排
        try:
            jobs = await cron.list_jobs("basic")
            for job in jobs:
                if getattr(job, "name", "") == "大肥鱼天气提醒":
                    await cron.delete_job(job.job_id)
        except Exception:  # noqa: BLE001 - 清理失败不影响新任务
            pass
        try:
            self._reminder_job = await add_job(
                name="大肥鱼天气提醒",
                cron_expression=expression,
                handler=self._reminder_tick,
                description="大肥鱼天气主动提醒",
                timezone=str(self._cfg("reminder.timezone", "") or "").strip()
                or None,
                persistent=False,
            )
            logger.info(
                "[%s] 主动提醒已排期：%s", PLUGIN_DISPLAY_NAME, expression
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 主动提醒排期失败：%s", PLUGIN_DISPLAY_NAME, exc)

    async def _remove_reminder_job(self) -> None:
        job = self._reminder_job
        self._reminder_job = None
        if job is None:
            return
        cron = getattr(self.context, "cron_manager", None)
        delete_job = getattr(cron, "delete_job", None)
        if delete_job is None:
            return
        try:
            await delete_job(job.job_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 取消主动提醒失败：%s", PLUGIN_DISPLAY_NAME, exc)

    async def _reminder_tick(self) -> None:
        if not self._cfg_bool("reminder.enabled", False):
            return
        now = datetime.now()
        quiet_start = max(0, min(24, self._cfg_int("reminder.quiet_start", 22)))
        quiet_end = max(0, min(24, self._cfg_int("reminder.quiet_end", 8)))
        if not (quiet_end <= now.hour < quiet_start):
            return
        today = now.strftime("%Y-%m-%d")
        temp_drop = self._cfg_int("reminder.temp_drop", 6)
        bindings = await self.store.all()
        for user_id, record in bindings.items():
            if record.get("last_reminder") == today:
                continue
            location = core.Location.from_dict(record.get("location") or {})
            if not location.name:
                continue
            try:
                data = await self.service.get(location, force_refresh=True)
            except core.WeatherError as exc:
                logger.info(
                    "[%s] 提醒前查天气失败（%s）：%s",
                    PLUGIN_DISPLAY_NAME,
                    user_id,
                    exc,
                )
                continue
            reason = core.reminder_reason(
                data, record.get("snapshot"), temp_drop
            )
            if not reason:
                continue
            umo = str(record.get("umo") or "").strip()
            platform = str(record.get("platform") or "aiocqhttp").strip()
            if ":GroupMessage:" in umo and platform:
                # 主动提醒尽量走私聊，不往群里发
                umo = f"{platform}:FriendMessage:{user_id}"
            if not umo:
                umo = f"{platform}:FriendMessage:{user_id}"
            text = await self._reminder_text(
                umo, str(record.get("nickname") or ""), data, reason
            )
            try:
                await StarTools.send_message(umo, MessageChain().message(text))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] 主动提醒发送失败（%s）：%s",
                    PLUGIN_DISPLAY_NAME,
                    user_id,
                    exc,
                )
                continue
            record["last_reminder"] = today
            record["snapshot"] = {
                "city": data.location.name,
                "temp": data.current_temp,
                "high": data.today_max,
                "low": data.today_min,
                "text": data.current_text,
                "ts": time.time(),
            }
            await self.store.set(user_id, record)
            logger.info(
                "[%s] 已主动提醒 %s（%s）", PLUGIN_DISPLAY_NAME, user_id, reason
            )

    async def _reminder_text(
        self,
        umo: str,
        nickname: str,
        data: core.WeatherData,
        reason: str,
    ) -> str:
        fallback = core.reminder_template(data, reason)
        if str(self._cfg("reply.mode", "llm")) != "llm":
            return fallback
        persona = await self._persona_prompt_for(umo)
        prompt = core.build_reminder_prompt(
            data,
            reason,
            nickname=nickname,
            max_chars=max(60, self._cfg_int("reply.max_chars", 200)),
            extra=str(self._cfg("reply.extra_prompt", "") or ""),
        )
        reply = core.clean_llm_reply(
            await self._llm_for(umo, persona, prompt),
            self._cfg_int("reply.max_chars", 200),
        )
        return reply or fallback

    # ------------------------------------------------------------------
    # LLM 工具（人格提示词里提到的 get_weather）
    # ------------------------------------------------------------------
    @filter.llm_tool(name="get_weather")
    async def get_weather_tool(self, event: AstrMessageEvent, city: str = "") -> str:
        """查询某座城市的实时天气和未来几天预报。

        用户问天气、下雨、下雪、冷热、带伞、空气质量时调用。

        Args:
            city(string): 城市名称，例如“杭州”“北京”。留空表示使用用户已经绑定的城市。
        """
        sender = str(event.get_sender_id() or "")
        location = None
        if city.strip():
            location = await self.service.resolve(city.strip())
        if location is None:
            binding = await self.store.get(sender)
            if binding:
                location = core.Location.from_dict(binding.get("location") or {})
        if location is None:
            default_city = str(self._cfg("basic.default_city", "") or "").strip()
            if default_city:
                location = await self.service.resolve(default_city)
        if location is None:
            return (
                "还没有记录这个用户所在的城市。"
                "请先用一句自然的、符合人设的话问问他现在在哪个城市，"
                "不要提“绑定”“设置”“系统”。"
            )
        try:
            data = await self.service.get(location)
        except core.WeatherError as exc:
            return f"天气源暂时不可用：{exc}"

        return (
            data.to_prompt_text(
                max_days=self._cfg_int("reply.forecast_days", 3),
                show_alerts=self._cfg_bool("features.show_alerts", True),
                show_air=self._cfg_bool("features.show_air_quality", True),
            )
            + "\n请用你平时说话的口吻转述给用户：1到3行，每行不超过30个字，"
            "不用句号，不要罗列字段。"
        )

    # ------------------------------------------------------------------
    # 可视面板 Web API
    # ------------------------------------------------------------------
    def _register_web_api(self) -> None:
        routes: list[tuple[str, Any, list[str], str]] = [
            ("/state", self.api_state, ["GET"], "天气插件状态"),
            ("/bindings/delete", self.api_binding_delete, ["POST"], "删除绑定"),
            ("/bindings/add", self.api_binding_add, ["POST"], "新增/修改绑定"),
            ("/cache/clear", self.api_cache_clear, ["POST"], "清空缓存"),
            ("/test", self.api_test, ["POST"], "测试查询"),
        ]
        register = getattr(self.context, "register_web_api", None)
        if register is None:
            logger.info("[%s] 当前 AstrBot 不支持插件 Web API，跳过面板", PLUGIN_DISPLAY_NAME)
            return
        for route, handler, methods, desc in routes:
            try:
                register(f"/{PLUGIN_NAME}{route}", handler, methods, desc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] 注册 Web API 失败：%s", PLUGIN_DISPLAY_NAME, exc)

    async def api_state(self):
        bindings = await self.store.all()
        rows = []
        for user_id, record in bindings.items():
            loc = core.Location.from_dict(record.get("location") or {})
            updated = record.get("updated")
            rows.append(
                {
                    "user_id": user_id,
                    "nickname": record.get("nickname") or "",
                    "city": loc.name,
                    "province": loc.province,
                    "display": loc.display,
                    "updated": _fmt_timestamp(updated),
                }
            )
        rows.sort(key=lambda item: item.get("updated") or "", reverse=True)
        return json_response(
            {
                "status": "ok",
                "data": {
                    "plugin": PLUGIN_NAME,
                    "display_name": PLUGIN_DISPLAY_NAME,
                    "sources": self.service.source_state(),
                    "bindings": rows,
                    "stats": dict(self.service.stats),
                    "cache": self.service.cache_size(),
                    "pending": len(self._pending),
                    "config": {
                        "reply_mode": self._cfg("reply.mode", "llm"),
                        "ask_location": self._cfg_bool("basic.ask_location", True),
                        "group_require_at": self._cfg_bool(
                            "basic.group_require_at", True
                        ),
                        "allow_private_all": self._cfg_bool(
                            "basic.allow_private_all", True
                        ),
                        "allowed_users": sorted(self._allowed_users),
                        "default_city": self._cfg("basic.default_city", ""),
                        "forecast_days": self._cfg_int("reply.forecast_days", 3),
                    },
                },
            }
        )

    async def api_binding_delete(self):
        payload = await request.json(default={}) or {}
        user_id = str(payload.get("user_id") or "").strip()
        if not user_id:
            return error_response("缺少 user_id", status_code=400)
        existed = await self.store.delete(user_id)
        return json_response({"status": "ok", "data": {"deleted": existed}})

    async def api_binding_add(self):
        payload = await request.json(default={}) or {}
        user_id = str(payload.get("user_id") or "").strip()
        city = str(payload.get("city") or "").strip()
        if not user_id or not city:
            return error_response("缺少 user_id 或 city", status_code=400)
        location = await self.service.resolve(city, force=True)
        if location is None:
            return error_response(f"没找到「{city}」这个地方", status_code=404)
        record = await self.store.get(user_id) or {}
        record.update(
            {
                "location": location.to_dict(),
                "updated": time.time(),
                "nickname": record.get("nickname", ""),
                "platform": record.get("platform", ""),
            }
        )
        await self.store.set(user_id, record)
        return json_response(
            {
                "status": "ok",
                "data": {"user_id": user_id, "display": location.display},
            }
        )

    async def api_cache_clear(self):
        self.service.clear_cache()
        return json_response({"status": "ok", "data": {"cache": 0}})

    async def api_test(self):
        payload = await request.json(default={}) or {}
        city = str(payload.get("city") or "").strip()
        if not city:
            return error_response("缺少 city", status_code=400)
        location = await self.service.resolve(city, force=True)
        if location is None:
            return error_response(f"没找到「{city}」这个地方", status_code=404)
        try:
            data = await self.service.get(location, force_refresh=True)
        except core.WeatherError as exc:
            return error_response(str(exc), status_code=502)
        return json_response(
            {
                "status": "ok",
                "data": {
                    "weather": data.to_dict(),
                    "template": core.render_template(
                        data,
                        "today",
                        set(),
                        show_air=self._cfg_bool(
                            "features.show_air_quality", True
                        ),
                    ),
                    "prompt": data.to_prompt_text(3),
                },
            }
        )
