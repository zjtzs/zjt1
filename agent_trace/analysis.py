# -*- coding: utf-8 -*-
"""指标提取层: 行为统计 / 阶段归类 / 校准后错误检测 / 循环检测。

约定 (经真实 297 条校准):
- 错误检测行级锚定: error:/FAILED/fatal:/critical 只认行首
- READ 语义工具的结果整体豁免 (文件内容含 error 字样是结构性误报)
- 数值误差指标行 ("conversion error: ...") 是求解器进度, 豁免
- 循环检测区分 kind: 轮询等待 (sleep/tail 等, 有意) vs 疑似重试
- 阶段归类为启发式, 产出是分析线索而非绝对结论
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

from .schema import UnifiedTrace, Event, KIND_TOOL_RESULT
from .semantic import (EXEC, READ, SEARCH, EDIT, WRITE, semantic_for_call,
                       classify_shell_command, tag_for_tool)

# ---- 错误检测规则 (全部行首锚定) ----
_RE_LINE_ERR = re.compile(
    r"^(?:error|failed|fatal|critical)\b.*", re.M | re.I)
_RE_EXIT_CODE = re.compile(r"^Exit code (\d+)\s*$", re.M)
_RE_NOFILE = re.compile(r"^(?:\S*: )?(?:No such file or directory|command not found)", re.M)
_RE_PERM = re.compile(r"^(?:\S*: )?Permission denied", re.M)
_RE_MODNOTFOUND = re.compile(r"^ModuleNotFoundError(?![\w])", re.M)
# 求解器进度/数值误差行豁免 (出现在行中且带数值)
_RE_NUMERIC_METRIC = re.compile(r"^\s*\w*error\w*\s*[:=]\s*-?[\d.eE+]+", re.I)

_POLL_TOOLS = {"sleep", "tail", "write_stdin", "watch", "read"}


def detect_errors(trace: UnifiedTrace, tag_by_seq: Dict[int, str]) -> List[Dict]:
    """行级错误检测。返回 [{seq, rule, line, excerpt}]。"""
    out: List[Dict] = []
    for e in trace.tool_results():
        if tag_by_seq.get(e.seq) == READ:
            continue  # READ 结果整体豁免
        # miniswe 结果的结构化 returncode (最可靠信号)
        rc = e.meta.get("returncode")
        if rc is not None and rc != 0:
            out.append({"seq": e.seq, "rule": "exit_code", "line": f"returncode={rc}",
                        "excerpt": (e.text or "")[:400], "code": int(rc)})
        text = e.text or ""
        seen_lines = set()
        for m in _RE_LINE_ERR.finditer(text):
            line = m.group(0).strip()
            if _RE_NUMERIC_METRIC.match(line):
                continue  # 数值误差指标 = 求解器进度
            out.append(_err(e, "line_error", line))
            seen_lines.add(line)
        for m in _RE_EXIT_CODE.finditer(text):
            code = int(m.group(1))
            if code != 0:
                out.append(_err(e, "exit_code", m.group(0), extra={"code": code}))
        for rule, rx in (("no_such_file", _RE_NOFILE), ("permission_denied", _RE_PERM),
                         ("module_not_found", _RE_MODNOTFOUND)):
            for m in rx.finditer(text):
                out.append(_err(e, rule, m.group(0).strip()))
    # 去重 (同 seq 同 rule 同 line)
    uniq, seen = [], set()
    for err in out:
        k = (err["seq"], err["rule"], err["line"])
        if k not in seen:
            seen.add(k)
            uniq.append(err)
    return uniq


def _err(e: Event, rule: str, line: str, extra: Optional[Dict] = None) -> Dict:
    d = {"seq": e.seq, "rule": rule, "line": line[:200],
         "excerpt": (e.text or "")[:400]}
    if extra:
        d.update(extra)
    return d


def detect_loops(pairs: List[Dict]) -> List[Dict]:
    """连续同参调用 ≥3 次 → 循环。区分轮询等待 vs 疑似重试。

    轮询命令 (sleep/tail/watch/write_stdin) 参数常逐次变化, 按命令首词归组;
    其余按 (工具名, 参数) 精确匹配。
    """
    loops = []
    run: List[Dict] = []

    def sig(p: Dict):
        name = p["call"].name
        cmd = str(p["call"].args.get("command", p["call"].args.get("cmd", "")))
        first = cmd.strip().split(" ", 1)[0] if cmd.strip() else ""
        if first in _POLL_TOOLS or name.lower() in _POLL_TOOLS:
            return (name, first or name)
        return (name, p["call"].arguments)

    for p in pairs:
        s = sig(p)
        if run and sig(run[-1]) == s:
            run.append(p)
        else:
            if len(run) >= 3:
                loops.append(_loop(run))
            run = [p]
    if len(run) >= 3:
        loops.append(_loop(run))
    return loops


def _loop(run: List[Dict]) -> Dict:
    cmd = run[0]["call"].args.get("command", "") or run[0]["call"].arguments
    is_poll = any(t in str(cmd).lower() for t in _POLL_TOOLS) or \
        run[0]["call"].name.lower() in _POLL_TOOLS
    return {"kind": "polling" if is_poll else "retry",
            "tool": run[0]["call"].name, "count": len(run),
            "start_seq": run[0]["call_seq"], "end_seq": run[-1]["result_seq"] or run[-1]["call_seq"],
            "sample_args": str(cmd)[:200]}


# ---- 阶段归类 ----
def stage_of(tag: str, args: Dict, result_text: str) -> str:
    """EXPLORE / BUILD / VERIFY 启发式。"""
    if tag in (READ, SEARCH):
        return "EXPLORE"
    if tag in (WRITE, EDIT):
        return "BUILD"
    if tag == EXEC:
        cmd = str(args.get("command", args.get("cmd", ""))).lower()
        if any(k in cmd for k in ("pytest", "unittest", "test", "check", "eval", "run ")):
            return "VERIFY"
        return "BUILD" if tag == EXEC and cmd.startswith(("python", "pytest")) else "EXPLORE"
    return "OTHER"


def empty_analysis(trace_id: str = "", reason: str = "") -> Dict:
    return {"trace_id": trace_id, "trajectory_available": False, "reason": reason,
            "n_events": 0, "tool_calls": 0, "paired": 0, "unpaired": 0,
            "pair_rate": 0.0, "semantic_counts": {}, "raw_tool_counts": {},
            "stage_counts": {}, "errors": [], "loops": [],
            "messages": {}, "usage": {}, "timestamps": {}}


def analyze_trace(trace: UnifiedTrace) -> Dict:
    # runner_exception 等运行层事件不算可用轨迹 (DSH 失败轨迹 messages=[] 但有 harness_events)
    if not trace.trajectory_available:
        a = empty_analysis(trace.trace_id, reason="trajectory_unavailable")
        a.update({"reward": trace.reward, "cost_usd": trace.cost_usd,
                  "agent_seconds": trace.agent_seconds,
                  "harness_events": trace.harness_events})
        return a

    pairs = trace.call_result_pairs()
    tag_by_seq: Dict[int, str] = {}
    semantic_counts: Counter = Counter()
    raw_counts: Counter = Counter()
    stage_counts: Counter = Counter()
    for p in pairs:
        args = p["call"].args
        tag, _ = semantic_for_call(p["call"].name, args)
        tag_by_seq[p["call_seq"]] = tag
        semantic_counts[tag] += 1
        raw_counts[p["call"].name] += 1
        res_text = p["result"].text if p["result"] else ""
        stage_counts[stage_of(tag, args, res_text)] += 1

    msg_counter = Counter(e.kind for e in trace.events)
    paired = sum(1 for p in pairs if p["result"] is not None)
    ts_list = [e.timestamp for e in trace.events if e.timestamp]
    ts_list.sort()
    a = {
        "trace_id": trace.trace_id,
        "trajectory_available": True,
        "n_events": len(trace.events),
        "tool_calls": len(pairs),
        "paired": paired,
        "unpaired": len(pairs) - paired,
        "pair_rate": round(paired / len(pairs), 4) if pairs else 0.0,
        "semantic_counts": dict(semantic_counts),
        "raw_tool_counts": dict(raw_counts),
        "stage_counts": dict(stage_counts),
        "errors": detect_errors(trace, tag_by_seq),
        "loops": detect_loops(pairs),
        "messages": dict(msg_counter),
        "usage": trace.usage,
        "reward": trace.reward,
        "cost_usd": trace.cost_usd,
        "agent_seconds": trace.agent_seconds,
        "timestamps": {"first": ts_list[0] if ts_list else "",
                       "last": ts_list[-1] if ts_list else "",
                       "coverage": round(len(ts_list) / len(trace.events), 3) if trace.events else 0.0},
    }
    return a
