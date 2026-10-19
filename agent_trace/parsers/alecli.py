# -*- coding: utf-8 -*-
"""ALE-CLI 信封解析器 (benchmark=ALE-CLI, 共享 OpenAI 风格消息结构)。

已知 harness: "Claude Code" (claude-opus-5) / "DSH" (deepseek-v4-pro) /
gpt-6-astra (exec 系)。三者消息字段一致:
  role/content/reasoning/tool_calls[{id,name,arguments}]/tool_call_id/name/timestamp
差异点:
- DSH 有 system 消息、subagents 字段; 失败轨迹 messages=[] 且
  harness_events=[{type: runner_exception, ...}]
- Claude Code 无 system; 并行 tool_calls (一条 assistant 多个调用)
- gpt 时间戳覆盖 ~93% (时间线必须用 seq 兜底)
"""
from __future__ import annotations

import os

from ..schema import (Event, ToolCall, UnifiedTrace, KIND_ASSISTANT,
                      KIND_RUNNER, KIND_SYSTEM, KIND_TOOL_RESULT, KIND_USER)
from . import BaseParser, register

MAX_TEXT = 20_000  # 超长文本截断


def _clip(text) -> str:
    if not isinstance(text, str):
        if text is None:
            return ""
        text = str(text)
    if len(text) > MAX_TEXT:
        return text[:MAX_TEXT]
    return text


class AleCliParser(BaseParser):
    name = "alecli"

    def parse(self, raw: dict, path: str = "", model_key: str = "") -> UnifiedTrace:
        stem = os.path.splitext(os.path.basename(path))[0]
        trace = UnifiedTrace(
            trace_id=stem,
            model_key=model_key or os.path.basename(os.path.dirname(path)),
            harness=str(raw.get("harness", "unknown")),
            task_id=str(raw.get("task_id", "")),
            source_path=os.path.normcase(os.path.abspath(path)),
            reward=_as_float(raw.get("reward")),
            reward_status=str(raw.get("reward_status", "")),
            cost_usd=_as_float(raw.get("cost_usd")) or 0.0,
            agent_seconds=_as_float(raw.get("agent_seconds")) or 0.0,
            usage=raw.get("usage") or {},
            trajectory_available=bool(raw.get("trajectory_available", True)),
            subagents=raw.get("subagents") or [],
            harness_events=raw.get("harness_events") or [],
            meta={
                "benchmark": raw.get("benchmark", ""),
                "model": raw.get("model", ""),
                "model_id": raw.get("model_id", ""),
                "harness_version": raw.get("harness_version", ""),
                "run_id": (raw.get("source") or {}).get("run_id", ""),
                "trial": (raw.get("source") or {}).get("trial", ""),
            },
        )
        events: list = []
        seq = 0
        for m in raw.get("messages") or []:
            role = m.get("role", "")
            content = m.get("content")
            reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
            ts = m.get("timestamp") or ""
            if role == "system":
                events.append(Event(seq=seq, kind=KIND_SYSTEM, text=_clip(content),
                                    timestamp=ts))
            elif role == "user":
                events.append(Event(seq=seq, kind=KIND_USER, text=_clip(content),
                                    timestamp=ts))
            elif role == "assistant":
                tcs = []
                for tc in m.get("tool_calls") or []:
                    args = tc.get("arguments", "")
                    if not isinstance(args, str):
                        import json as _json
                        try:
                            args = _json.dumps(args, ensure_ascii=False)
                        except (TypeError, ValueError):
                            args = str(args)
                    tcs.append(ToolCall(call_id=str(tc.get("id", "")),
                                        name=str(tc.get("name", "")),
                                        arguments=args))
                events.append(Event(seq=seq, kind=KIND_ASSISTANT, text=_clip(content),
                                    reasoning=_clip(reasoning), tool_calls=tcs,
                                    timestamp=ts))
            elif role == "tool":
                events.append(Event(seq=seq, kind=KIND_TOOL_RESULT, text=_clip(content),
                                    tool_call_id=str(m.get("tool_call_id", "")),
                                    tool_name=str(m.get("name", "")), timestamp=ts))
            else:
                events.append(Event(seq=seq, kind=KIND_RUNNER, text=_clip(content),
                                    meta={"role": role}))
            seq += 1
        # harness 层事件 (runner_exception 等) 附加为 RUNNER 事件
        for he in trace.harness_events:
            events.append(Event(seq=seq, kind=KIND_RUNNER,
                                text=str(he.get("detail", "")),
                                meta={"type": he.get("type", ""),
                                      "index": he.get("index", 0)}))
            seq += 1
        trace.events = events
        return trace


def _as_float(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


@register("Claude Code")
class ClaudeCodeParser(AleCliParser):
    name = "claude-code"


@register("DSH")
class DshParser(AleCliParser):
    name = "dsh"


@register("__ale_generic__")
class AleGenericParser(AleCliParser):
    """未知 harness 名的 ALE-CLI 信封兜底 (如 gpt-6-astra 的 harness 名)。"""
    name = "ale-generic"
