# -*- coding: utf-8 -*-
"""mini-swe-agent SWE-bench 轨迹解析器 ({model}/{instance_id}/{instance_id}.traj.json)。

结构要点 (真实数据勘察):
- assistant.tool_calls: [{id, type:'function', function:{name, arguments}}] (OpenAI 嵌套)
- reasoning 字段双形态: claude-4-5 → reasoning_content(+thinking_blocks); glm-5 → reasoning
- tool 结果: content="<returncode>N</returncode>\\n<output>...</output>",
  extra.{raw_output, returncode, timestamp(unix float), exception_info}; 不回显工具名
  (解析时用 call_id→name 映射回填; mini-swe-agent 只用 bash)
- exit 消息: content=最终 patch 或 "LimitsExceeded", extra.{exit_status, submission}
- info: model_stats{instance_cost, api_calls} / config.model.model_name / exit_status / submission
- resolved 标注由 store 层从 per_instance_details__{model}.json 注入 meta.benchmark
"""
from __future__ import annotations

import os

from ..schema import (Event, ToolCall, UnifiedTrace, KIND_ASSISTANT,
                      KIND_RUNNER, KIND_SYSTEM, KIND_TOOL_RESULT, KIND_USER)
from . import BaseParser, register

MAX_TEXT = 20_000


def _clip(text) -> str:
    if not isinstance(text, str):
        if text is None:
            return ""
        text = str(text)
    return text[:MAX_TEXT]


def _detect(raw: dict) -> bool:
    return (isinstance(raw.get("instance_id"), str)
            and isinstance(raw.get("messages"), list)
            and isinstance(raw.get("info"), dict)
            and "exit_status" in (raw.get("info") or {}))


@register("minisweagent", detector=_detect)
class MiniSweParser(BaseParser):
    name = "minisweagent"

    def parse(self, raw: dict, path: str = "", model_key: str = "") -> UnifiedTrace:
        instance_id = str(raw.get("instance_id", ""))
        info = raw.get("info") or {}
        cfg = info.get("config") or {}
        stats = info.get("model_stats") or {}
        events: list = []
        call_name: dict = {}  # call_id -> tool name (结果消息不回显工具名)
        ts_first = ts_last = None
        seq = 0

        for m in raw.get("messages") or []:
            role = m.get("role", "")
            content = m.get("content")
            extra = m.get("extra") or {}
            if role == "system":
                events.append(Event(seq=seq, kind=KIND_SYSTEM, text=_clip(content)))
            elif role == "user":
                events.append(Event(seq=seq, kind=KIND_USER, text=_clip(content)))
            elif role == "assistant":
                reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
                tcs = []
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    name = str(fn.get("name") or tc.get("name") or "")
                    args = fn.get("arguments", tc.get("arguments", ""))
                    if not isinstance(args, str):
                        args = str(args)
                    cid = str(tc.get("id", ""))
                    tcs.append(ToolCall(call_id=cid, name=name, arguments=args))
                    if cid:
                        call_name[cid] = name
                events.append(Event(seq=seq, kind=KIND_ASSISTANT, text=_clip(content),
                                    reasoning=_clip(reasoning), tool_calls=tcs))
            elif role == "tool":
                ts = extra.get("timestamp")
                if isinstance(ts, (int, float)):
                    ts_first = ts if ts_first is None else min(ts_first, ts)
                    ts_last = ts if ts_last is None else max(ts_last, ts)
                # returncode 提升到 meta, 供错误检测/归因直接使用
                rc = extra.get("returncode")
                meta = {}
                if rc is not None:
                    meta["returncode"] = rc
                if extra.get("exception_info"):
                    meta["exception_info"] = str(extra["exception_info"])[:500]
                events.append(Event(seq=seq, kind=KIND_TOOL_RESULT, text=_clip(content),
                                    tool_call_id=str(m.get("tool_call_id", "")),
                                    tool_name=call_name.get(str(m.get("tool_call_id", "")), "bash"),
                                    meta=meta))
            elif role == "exit":
                events.append(Event(seq=seq, kind=KIND_RUNNER, text=_clip(content),
                                    meta={"exit_status": extra.get("exit_status") or info.get("exit_status", ""),
                                          "role": "exit"}))
            else:
                events.append(Event(seq=seq, kind=KIND_RUNNER, text=_clip(content),
                                    meta={"role": role}))
            seq += 1

        agent_seconds = round(ts_last - ts_first, 1) if (ts_first and ts_last) else 0.0
        return UnifiedTrace(
            trace_id=instance_id,
            model_key=model_key or os.path.basename(os.path.dirname(os.path.dirname(path))),
            harness="minisweagent",
            task_id=instance_id,
            source_path=os.path.normcase(os.path.abspath(path)),
            events=events,
            cost_usd=float(stats.get("instance_cost") or 0.0),
            agent_seconds=agent_seconds,
            usage={"model_calls": int(stats.get("api_calls") or 0)},
            trajectory_available=True,
            meta={
                "benchmark": "swe-bench",
                "model": (cfg.get("model") or {}).get("model_name", ""),
                "mini_version": info.get("mini_version", ""),
                "exit_status": info.get("exit_status", ""),
                "submission": _clip(info.get("submission", "")),
                "agent_type": cfg.get("agent_type", ""),
                "environment_type": cfg.get("environment_type", ""),
            },
        )
