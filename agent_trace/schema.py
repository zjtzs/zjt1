# -*- coding: utf-8 -*-
"""统一中间表示 (IR): UnifiedTrace / Event / ToolCall。

所有 harness 的原始轨迹经解析器归一到本 IR，下游 (analysis/attribution/
dynamics/security/causal/standard) 只消费 IR，不接触原始格式。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """一次工具调用 (来自 assistant 消息的 tool_calls 数组元素)。"""

    call_id: str = ""
    name: str = ""          # 原始工具名 (异构: PascalCase / snake_case / exec 系)
    arguments: str = ""     # 原始 JSON 字符串

    _args_cache: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)

    @property
    def args(self) -> Dict[str, Any]:
        """arguments 的 dict 视图；解析失败返回 {} (参数可能是非 JSON 文本)。"""
        if self._args_cache is None:
            try:
                v = json.loads(self.arguments) if self.arguments else {}
                self._args_cache = v if isinstance(v, dict) else {}
            except (ValueError, TypeError):
                self._args_cache = {}
        return self._args_cache

    def to_dict(self) -> Dict[str, Any]:
        return {"call_id": self.call_id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolCall":
        return cls(call_id=d.get("call_id", ""), name=d.get("name", ""),
                   arguments=d.get("arguments", ""))


# 事件种类
KIND_USER = "user"
KIND_SYSTEM = "system"
KIND_ASSISTANT = "assistant"      # 模型输出 (可含 tool_calls)
KIND_TOOL_RESULT = "tool_result"  # 工具/环境返回
KIND_RUNNER = "runner"            # harness 层事件 (如 runner_exception)


@dataclass
class Event:
    """轨迹中的一个事件。seq 为轨迹内单调递增序号 (时间戳缺失时的时间线兜底)。"""

    seq: int
    kind: str
    text: str = ""                  # 消息/结果正文 (超长截断到 20k, 截断信息在 meta)
    reasoning: str = ""             # 思考文本 (claude/DSH 有, gpt 系多数无)
    tool_calls: List[ToolCall] = field(default_factory=list)  # assistant 专用
    tool_call_id: str = ""          # tool_result 专用: 对应调用的 call_id
    tool_name: str = ""             # tool_result 回显的原始工具名
    timestamp: str = ""             # ISO 字符串, 允许缺失 (用 seq 兜底)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_tool_result(self) -> bool:
        return self.kind == KIND_TOOL_RESULT

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        return cls(
            seq=d.get("seq", 0),
            kind=d.get("kind", ""),
            text=d.get("text", ""),
            reasoning=d.get("reasoning", ""),
            tool_calls=[ToolCall.from_dict(c) for c in d.get("tool_calls", [])],
            tool_call_id=d.get("tool_call_id", ""),
            tool_name=d.get("tool_name", ""),
            timestamp=d.get("timestamp", ""),
            meta=d.get("meta", {}),
        )


@dataclass
class UnifiedTrace:
    """一条 agent 轨迹的统一表示。"""

    trace_id: str
    model_key: str              # 所属模型目录名 (store 键, 如 "claude-opus-5_lines")
    harness: str                # 归一前的 harness 名 (如 "Claude Code" / "DSH" / "minisweagent")
    task_id: str
    source_path: str            # 原始文件路径 (store 缓存主键, 已 normcase)
    events: List[Event] = field(default_factory=list)
    reward: Optional[float] = None
    reward_status: str = ""
    cost_usd: float = 0.0
    agent_seconds: float = 0.0
    usage: Dict[str, Any] = field(default_factory=dict)
    trajectory_available: bool = True
    subagents: List[Dict[str, Any]] = field(default_factory=list)
    harness_events: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)  # model/model_id/version/benchmark/... 及 harness 专有信息

    # ---------- 序列化 ----------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnifiedTrace":
        # 关键: 先浅拷贝并 pop events, 避免构造器与 d["events"] 共享同一 list 引用
        # (经典别名 bug: 循环 append 到正在迭代的列表)。
        d = dict(d)
        events = [Event.from_dict(e) for e in d.pop("events", [])]
        return cls(events=events, **{k: v for k, v in d.items() if k in cls._fields()})

    @classmethod
    def _fields(cls) -> set:
        return {f for f in cls.__dataclass_fields__ if f != "events"}

    # ---------- 派生视图 ----------
    def text_events(self) -> List[Event]:
        """assistant 纯文本事件 (content 非空的 assistant 消息)。"""
        return [e for e in self.events if e.kind == KIND_ASSISTANT and e.text]

    def call_result_pairs(self) -> List[Dict[str, Any]]:
        """调用-结果配对。

        优先按 tool_call_id 精确配对; 结果缺失或 id 缺失时按 FIFO 队列兜底。
        返回 [{"call": ToolCall, "call_seq": int, "result": Event|None, "result_seq": int|None}]
        result 可为 None (调用未获回应, 如截断轨迹)。
        """
        by_id: Dict[str, Dict[str, Any]] = {}
        fifo: List[Dict[str, Any]] = []
        pairs: List[Dict[str, Any]] = []
        for e in self.events:
            if e.kind == KIND_ASSISTANT:
                for tc in e.tool_calls:
                    p = {"call": tc, "call_seq": e.seq, "result": None, "result_seq": None}
                    pairs.append(p)
                    if tc.call_id:
                        by_id[tc.call_id] = p
                    else:
                        fifo.append(p)
            elif e.kind == KIND_TOOL_RESULT:
                p = by_id.pop(e.tool_call_id, None) if e.tool_call_id else None
                if p is None:
                    p = fifo.pop(0) if fifo else None
                if p is not None:
                    p["result"] = e
                    p["result_seq"] = e.seq
        return pairs

    def tool_results(self) -> List[Event]:
        return [e for e in self.events if e.kind == KIND_TOOL_RESULT]
