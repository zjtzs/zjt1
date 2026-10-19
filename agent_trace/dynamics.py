# -*- coding: utf-8 -*-
"""进展动力学: 单轨迹内的进展信号、冻结窗口与错误谱系。

进展信号三来源 (校准自真实数据):
1. 产物步: WRITE/EDIT 成功, 或 EXEC 带写文件形态 (heredoc/tee/重定向/touch 系) 成功
   - codex 系几乎不用 Write/Edit, 文件全靠 exec+heredoc 写, 必须覆盖
   - 2>/dev/null 只是 stderr 静默, 不是状态改变 (负向断言已在 semantic 层处理)
2. 修复确认: 同 tool+同参数, 前次报错、本次干净
3. 测试进展: 同测试命令, 前败后成 (参数可不同)

指标: 冻结占比 (最后进展步之后的游程占比) / 零状态变化占比 / 错误谱系。
注意: 最后一步恰为进展步时 stall 为空, 下游访问前必须判空。
"""
from __future__ import annotations

import re
from typing import Dict, List

from .schema import UnifiedTrace
from .semantic import WRITE, EDIT, EXEC, semantic_for_call, classify_shell_command
from .analysis import detect_errors

# 测试命令识别
_RE_TEST_CMD = re.compile(r"\b(pytest|unittest|python\s+-m\s+unittest|npm\s+test|go\s+test|cargo\s+test|make\s+test|tox)\b", re.I)
_RE_TEST_FAIL_LINE = re.compile(r"^\s*(?:FAILED|ERROR)\b|^\s*failing|^\s*\d+ failed\b", re.M | re.I)
_RE_TEST_PASS_LINE = re.compile(r"\b(?:passed|ok)\b", re.I)


def _is_clean(result) -> bool:
    """结果是否'干净' (无错误信号)。"""
    if result is None:
        return False
    rc = result.meta.get("returncode")
    if rc is not None and rc != 0:
        return False
    text = result.text or ""
    m = re.search(r"^Exit code (\d+)\s*$", text, re.M)
    if m and int(m.group(1)) != 0:
        return False
    if _RE_TEST_FAIL_LINE.search(text):
        return False
    if re.search(r"^(?:error|fatal|critical)\b", text, re.M | re.I):
        return False
    return True


def _has_error_signal(result) -> bool:
    return not _is_clean(result) and result is not None


def analyze_dynamics(trace: UnifiedTrace) -> Dict:
    pairs = trace.call_result_pairs()
    if not pairs:
        return {"available": False, "reason": "no_tool_calls" if trace.events else "trajectory_unavailable",
                "n_steps": 0, "steps": [], "progress_seqs": [], "frozen": {},
                "error_spectrum": {}, "metrics": {}}

    steps: List[Dict] = []
    progress_seqs: List[int] = []
    seen_err: Dict[str, bool] = {}       # 同 tool+args 上一轮是否有错
    test_seen_fail: bool = False          # 测试命令上一轮是否失败

    for p in pairs:
        call, result = p["call"], p["result"]
        tag, _ = semantic_for_call(call.name, call.args)
        cmd = str(call.args.get("command", call.args.get("cmd", "")) or "")
        args_key = call.name + "|" + call.arguments
        step = {"seq": p["call_seq"], "result_seq": p["result_seq"], "tool": call.name,
                "tag": tag, "cmd": cmd[:160], "kind": "action", "progress": False,
                "error": False, "test": False}
        if result is None:
            step["kind"] = "no_result"
            steps.append(step)
            continue

        clean = _is_clean(result)
        err = _has_error_signal(result)
        step["error"] = err
        # 测试命令?
        is_test = bool(_RE_TEST_CMD.search(cmd)) or bool(_RE_TEST_CMD.search(call.name))
        step["test"] = is_test
        # 进展判定
        prog = False
        if clean:
            if tag in (WRITE, EDIT):
                prog = True                     # 产物步
            elif tag == EXEC:
                stag, _ = classify_shell_command(cmd)
                if stag == WRITE:
                    prog = True                 # exec+heredoc 写文件成功
            if is_test and not _RE_TEST_FAIL_LINE.search(result.text or ""):
                if test_seen_fail:
                    prog = True                 # 测试前败后成
                step["kind"] = "test"
            if not prog and seen_err.get(args_key) and err is False:
                prog = True                     # 同参修复确认
                step["kind"] = "fix"
        elif is_test:
            step["kind"] = "test"
        # 更新状态记忆
        if is_test:
            test_seen_fail = err or bool(_RE_TEST_FAIL_LINE.search(result.text or ""))
        if tag == EXEC and cmd:
            pass
        seen_err[args_key] = err
        step["progress"] = prog
        if prog:
            progress_seqs.append(p["call_seq"])
        steps.append(step)

    # 冻结窗口: 最后一个进展步之后的游程 (按事件序号)
    if progress_seqs:
        prog_res_seq = {s["seq"]: (s["result_seq"] or s["seq"]) for s in steps if s["progress"]}
        last_prog_end = max(prog_res_seq.values())
        frozen_events = [e for e in trace.events if e.seq > last_prog_end]
    else:
        frozen_events = list(trace.events)  # 全程无进展 → 整段冻结
    n_total = len(trace.events)
    frozen_ratio = round(len(frozen_events) / n_total, 4) if n_total else 0.0
    n_progress = len(progress_seqs)
    zero_change = sum(1 for s in steps if not s["progress"] and not s["error"]
                      and s["kind"] in ("action", "no_result"))
    zero_ratio = round(zero_change / len(steps), 4) if steps else 0.0

    # 错误谱系: 只记调用侧游程, last_line 存命中错误行而非结果首行
    spectrum: Dict[str, int] = {}
    last_err_line = ""
    err_by_result_seq = {}
    for e in detect_errors(trace, {s["seq"]: s["tag"] for s in steps}):
        err_by_result_seq.setdefault(e["seq"], []).append(e)
        spectrum[e["rule"]] = spectrum.get(e["rule"], 0) + 1
        last_err_line = e["line"]

    return {
        "available": True,
        "n_steps": len(steps),
        "steps": steps,
        "progress_seqs": progress_seqs,
        "frozen": {
            "n_events": len(frozen_events),
            "ratio": frozen_ratio,
            "start_seq": (frozen_events[0].seq if frozen_events else None),
            "has_progress": bool(progress_seqs),
        },
        "metrics": {
            "n_progress": n_progress,
            "progress_ratio": round(n_progress / len(steps), 4) if steps else 0.0,
            "zero_change_ratio": zero_ratio,
            "n_errors": sum(spectrum.values()),
        },
        "error_spectrum": spectrum,
        "last_error_line": last_err_line,
    }
