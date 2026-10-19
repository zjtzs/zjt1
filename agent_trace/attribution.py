# -*- coding: utf-8 -*-
"""失败归因 v2: 只回答 reward<1 (或 resolved=False) 的"为什么", 判定看终局。

优先级 (经真实 297 条 + 合成数据双重校准):
  infra > env_missing > test_never_pass > stuck_loop > self_test_ok
  > gave_up_high > claimed_done > gave_up_low > no_tools > incomplete > unlocated

关键校准教训 (内置):
- 最大的失败模式是"未验证即宣告完成" → 完成动词集必须宽, 且 claimed_done
  判定先于 gave_up 低置信兜底
- ModuleNotFoundError 必须先于通用 py_exception 类规则 (env 优先级)
- 循环归属用结果序列上的游程标记 (analysis.detect_loops), 不用 seq 窗口
- 证据链必须提取命中行而非整段文本, 否则不可复核
- stall 可能为空 (最后一步恰是进展步), 访问 stall[-1] 前判空
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .analysis import analyze_trace, empty_analysis
from .dynamics import analyze_dynamics
from .schema import UnifiedTrace

# ---- 终局语言规则 (顺序敏感) ----
# 完成声明动词集 (宽)
_RE_CLAIM_DONE = re.compile(
    r"\b(?:done|completed|complete|finished|implemented|written|created|"
    r"validated|verified|fixed|solved|successfully|saved|generated|updated|"
    r"applied|added|ready|works|working|passes?|passed)\b", re.I)
# 高置信放弃语言
_RE_GAVE_UP = re.compile(
    r"\b(?:cannot|can't|unable to|impossible|give up|giving up|abandon|"
    r"stuck|no way|failed to (?:fix|solve|resolve|complete)|not able to)\b", re.I)
# 显式未完成声明
_RE_INCOMPLETE = re.compile(
    r"\b(?:not (?:yet )?(?:complete|finished|done)|still (?:need|working|remains)|"
    r"todo|next step|further work|incomplete)\b", re.I)

# env 类错误 (ModuleNotFoundError 在前 — 校准教训)
_RE_ENV = re.compile(
    r"^(?:ModuleNotFoundError|ImportError|conda|mamba|pip(?:3)?(?:\s|$)).*"
    r"|No module named|command not found|No such file or directory", re.M | re.I)

_RE_TEST_FAIL = re.compile(r"^\s*(?:FAILED|ERROR)\b|\b\d+ failed\b|\bfailed\b.*\btest", re.M | re.I)
_RE_TEST_PASS = re.compile(r"\b(?:\d+ passed|all tests? passed|ok)\b", re.I)


def _terminal_pairs(pairs: List[Dict], window: int = 5, ratio: float = 0.3) -> List[Dict]:
    """终局工具结果: 最后 window 个结果 + 轨迹后 ratio% 游程内的结果。"""
    with_res = [p for p in pairs if p["result"] is not None]
    if not with_res:
        return []
    n = len(with_res)
    tail_n = max(window, int(n * ratio))
    return with_res[-tail_n:]


def attribute(trace: UnifiedTrace, analysis: Optional[Dict] = None,
              dyn: Optional[Dict] = None) -> Dict:
    analysis = analysis or analyze_trace(trace)
    dyn = dyn or analyze_dynamics(trace)

    base = {"trace_id": trace.trace_id, "model_key": trace.model_key,
            "task_id": trace.task_id, "reward": trace.reward}

    # benchmark 标注: ALE 信封 meta.benchmark 是字符串 ("ALE-CLI"),
    # SWE-bench 轨迹是 dict {name, resolved, cost, api_calls} — 必须分支取值
    bm = trace.meta.get("benchmark")
    resolved = bm.get("resolved") if isinstance(bm, dict) else None
    base["resolved"] = resolved

    # ---- 成功不归因 ----
    success = (resolved is True) or (trace.reward is not None and trace.reward >= 1.0)
    if success:
        return {**base, "code": "success", "confidence": 1.0, "evidence": [],
                "frozen": dyn.get("frozen", {})}

    # ---- infra: 轨迹层失败 ----
    if not trace.trajectory_available:
        he = trace.harness_events or []
        ev = [{"seq": e.seq, "rule": "runner_exception", "line": e.text[:200]}
              for e in trace.events if e.kind == "runner"][:3]
        return {**base, "code": "infra", "confidence": 1.0, "evidence": ev,
                "detail": str(he[0].get("detail", "")) if he else "",
                "frozen": dyn.get("frozen", {})}

    pairs = trace.call_result_pairs()
    if not pairs:
        return {**base, "code": "no_tools", "confidence": 1.0, "evidence": [],
                "frozen": dyn.get("frozen", {})}

    term = _terminal_pairs(pairs)
    term_errors = [p for p in term if not _is_clean_res(p["result"])]
    last_res = term[-1]["result"] if term else None
    stall = dyn.get("frozen", {})
    frozen_events = stall.get("n_events", 0)

    # ---- env_missing: 终局窗口的环境类错误 (ModuleNotFound 优先) ----
    env_hits: List[Dict] = []
    for p in term:
        text = p["result"].text or ""
        for m in _RE_ENV.finditer(text):
            line = text[m.start():].splitlines()[0].strip()
            env_hits.append({"seq": p["result_seq"], "rule": "env_missing", "line": line[:200]})
    if env_hits and term_errors:
        # 终局以环境错误收尾才算 env; 若最后结果干净则不判 env
        if last_res is not None and not _is_clean_res(last_res):
            return {**base, "code": "env_missing", "confidence": 0.85,
                    "evidence": env_hits[:3], "frozen": stall}

    # ---- test_never_pass: 终局窗口测试在跑且从未通过 ----
    test_results = [p for p in term
                    if _RE_TEST_CMD_HINT.search(str(p["call"].args))]
    if test_results:
        last_test = test_results[-1]
        if not _is_clean_res(last_test["result"]) or _RE_TEST_FAIL.search(last_test["result"].text or ""):
            ev = [{"seq": last_test["result_seq"], "rule": "test_never_pass",
                   "line": (last_test["result"].text or "").strip().splitlines()[-1][:200]
                   if (last_test["result"].text or "").strip() else ""}]
            return {**base, "code": "test_never_pass", "confidence": 0.85,
                    "evidence": ev, "frozen": stall}
        # 测试最终通过但 reward<1 → 自测通过 (实现与评测预期不符)
        if _is_clean_res(last_test["result"]):
            return {**base, "code": "self_test_ok", "confidence": 0.8,
                    "evidence": [{"seq": last_test["result_seq"], "rule": "self_test_ok",
                                  "line": (last_test["result"].text or "").strip().splitlines()[-1][:200]}],
                    "frozen": stall}

    # ---- stuck_loop: 终局窗口存在疑似重试游程 (轮询不算) ----
    retry_loops = [lp for lp in analysis.get("loops", []) if lp["kind"] == "retry"]
    if retry_loops:
        lp = retry_loops[-1]
        return {**base, "code": "stuck_loop", "confidence": 0.8,
                "evidence": [{"seq": lp["start_seq"], "rule": "loop_retry",
                              "line": f"{lp['tool']} x{lp['count']}: {lp['sample_args']}"}],
                "frozen": stall}

    # ---- 纯文本收尾三分支 (完成声明先于放弃低置信 — 校准教训) ----
    texts = [e for e in trace.events if e.kind == "assistant" and e.text]
    tail_text = " ".join(t.text for t in texts[-2:]) if texts else ""
    if tail_text:
        if _RE_CLAIM_DONE.search(tail_text):
            # 未验证宣告完成: 终局无错误收尾/无测试失败信号
            return {**base, "code": "claimed_done", "confidence": 0.75,
                    "evidence": [{"seq": texts[-1].seq, "rule": "claim_done",
                                  "line": texts[-1].text[:200]}],
                    "frozen": stall}
        if _RE_GAVE_UP.search(tail_text):
            return {**base, "code": "gave_up_high", "confidence": 0.8,
                    "evidence": [{"seq": texts[-1].seq, "rule": "gave_up",
                                  "line": texts[-1].text[:200]}],
                    "frozen": stall}
        return {**base, "code": "gave_up_low", "confidence": 0.5,
                "evidence": [{"seq": texts[-1].seq, "rule": "text_terminal",
                              "line": texts[-1].text[:200]}],
                "frozen": stall}

    # ---- 未用工具之外: 实现未完成 ----
    if term_errors and frozen_events > 0:
        e0 = term_errors[-1]["result"]
        last_line = ""
        if e0 is not None:
            t_ = (e0.text or "").strip()
            last_line = t_.splitlines()[-1][:200] if t_ else ""
        return {**base, "code": "incomplete", "confidence": 0.6,
                "evidence": [{"seq": term_errors[-1]["result_seq"], "rule": "terminal_error",
                              "line": last_line}],
                "frozen": stall}
    return {**base, "code": "unlocated", "confidence": 0.3, "evidence": [],
            "frozen": stall}


_RE_TEST_CMD_HINT = re.compile(r"\b(?:pytest|unittest|npm test|go test|cargo test|tox)\b", re.I)


def _is_clean_res(result) -> bool:
    from .dynamics import _is_clean
    return result is not None and _is_clean(result)
