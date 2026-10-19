# -*- coding: utf-8 -*-
"""达标评测通道: 四硬门槛 (CCF PL 2026 海报目标) + 归因标注数据集导出。

门槛协议 (关键设计: 全部非循环):
  G1 一致性 — 三个成分, 分开报告, 不合并成单一数字:
    1a 可归因率: 失败轨迹中给出具体归因码 (非 unlocated/no_tools) 的比例 ≥85%。
    1b 成功特异度: success 判定仅由评估信号 (reward≥1 / resolved=True) 触发,
       与标注一致为结构保证 (by construction); 框架不猜测成功。
    1c 掩蔽探针 (discovery, 不计门禁): 掩蔽 resolved+reward 后框架独立判定 vs
       标注的一致率。原理性上界: "成功"由外部测试定义, 纯轨迹信号只能证伪
       (发现失败证据) 不能证实成功, 因此该一致率天然受限 — 如实报告并解释。
  G2 单例时延: 完整管线 (analysis+dynamics+attribution) 单例 ≤120s。
  G3 批量吞吐: 200 条 raw JSON 解析+全管线 ≤1200s (20min)。
  G4 成本: 本框架 $0 (纯 CPU 零 API); 对照 LLM-as-judge 估算基线, 要求 ≤50%。
"""
from __future__ import annotations

import copy
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Dict, List

from .analysis import analyze_trace
from .attribution import attribute
from .dynamics import analyze_dynamics
from .parsers import parse_file

# ---- 门槛阈值 ----
G1_ATTRIB_RATE_MIN = 0.85
G2_SECONDS_MAX = 120.0
G3_SECONDS_MAX = 1200.0
G4_RATIO_MAX = 0.5

# ---- LLM-as-judge 估算假设 (报告中明示) ----
JUDGE_PRICE_IN = 3.0    # $/M input tokens (主流中档模型口径)
JUDGE_PRICE_OUT = 15.0  # $/M output tokens
JUDGE_OUT_TOKENS = 300  # 判定+理由输出
CHARS_PER_TOKEN = 3.5   # 中英混合代码文本的经验值

UNLOCATED_CODES = {"unlocated", "no_tools"}
POSITIVE_CODES = {"success", "self_test_ok"}


def labeled_traces(traces: List) -> List:
    """带 resolved 人工标注的轨迹 (SWE-bench 200 条)。"""
    return [t for t in traces
            if isinstance(t.meta.get("benchmark"), dict)
            and t.meta["benchmark"].get("resolved") is not None]


def masked_attribute(trace) -> Dict:
    """掩蔽 resolved+reward 后独立归因 (不改动原对象)。"""
    t2 = copy.deepcopy(trace)
    bm = t2.meta.get("benchmark")
    if isinstance(bm, dict):
        bm["resolved"] = None
    t2.reward = None
    return attribute(t2)


def gate1_consistency(traces, attrs: Dict) -> Dict:
    fails = [(t, attrs[(t.model_key, t.trace_id)]) for t in traces
             if attrs[(t.model_key, t.trace_id)]["code"] != "success"]
    n_fail = len(fails)
    n_attributable = sum(1 for _, a in fails if a["code"] not in UNLOCATED_CODES)
    rate = n_attributable / n_fail if n_fail else None
    # 1c: 掩蔽探针 (标注轨迹) + self_test_ok 信号与 reward 的相关性 (ALE 有 reward)
    conf = defaultdict(int)
    probe = {}
    lat = labeled = traces
    for t in labeled_traces(traces):
        lab = t.meta["benchmark"]["resolved"]
        code = masked_attribute(t)["code"]
        conf[(lab, "pass" if code in POSITIVE_CODES else "fail")] += 1
    tp, fn = conf[(True, "pass")], conf[(True, "fail")]
    tn, fp = conf[(False, "fail")], conf[(False, "pass")]
    if (tp + fn + tn + fp):
        probe = {
            "n": tp + fn + tn + fp,
            "agreement": round((tp + tn) / (tp + fn + tn + fp), 4),
            "fail_detection": round(tn / (tn + fp), 4) if (tn + fp) else None,
            "success_recognition": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "confusion": {"TP": tp, "FN": fn, "TN": tn, "FP": fp},
            "note": ("掩蔽 resolved+reward 后的独立判定。成功由外部测试定义, 纯轨迹"
                     "信号只能证伪不能证实 (信息论上界), 该一致率为原理受限的诚实下界"),
        }
    ok = rate is not None and rate >= G1_ATTRIB_RATE_MIN
    return {"gate": "G1 一致性", "threshold": G1_ATTRIB_RATE_MIN, "passed": ok,
            "attributable_rate": round(rate, 4) if rate is not None else None,
            "n_failure_traces": n_fail, "n_attributable": n_attributable,
            "success_specificity": "by-construction (success 仅由评估信号触发)",
            "masked_probe": probe}


def gate2_latency(traces) -> Dict:
    lat = []
    for t in labeled_traces(traces):
        t0 = time.perf_counter()
        attribute(t, analysis=analyze_trace(t), dyn=analyze_dynamics(t))
        lat.append(time.perf_counter() - t0)
    lat.sort()
    n = len(lat)
    p50 = lat[n // 2] if n else 0.0
    p95 = lat[min(int(n * 0.95), n - 1)] if n else 0.0
    return {"gate": "G2 单例时延", "threshold_s": G2_SECONDS_MAX,
            "passed": bool(n) and p95 <= G2_SECONDS_MAX,
            "n": n, "p50_s": round(p50, 4), "p95_s": round(p95, 4),
            "max_s": round(lat[-1], 4) if n else 0.0}


def gate3_throughput(traces) -> Dict:
    """raw JSON → 解析 → 全管线, 200 条标注轨迹端到端。"""
    items = [(t.source_path, t.model_key) for t in labeled_traces(traces)]
    t0 = time.perf_counter()
    for path, mk in items:
        t = parse_file(path, model_key=mk)
        a = analyze_trace(t)
        d = analyze_dynamics(t)
        attribute(t, analysis=a, dyn=d)
    total = time.perf_counter() - t0
    return {"gate": "G3 批量吞吐", "threshold_s": G3_SECONDS_MAX,
            "passed": total <= G3_SECONDS_MAX,
            "n": len(items), "total_s": round(total, 2),
            "per_trace_s": round(total / max(len(items), 1), 4)}


def gate4_cost(traces) -> Dict:
    """本框架 $0 vs LLM-as-judge 估算 (输入=轨迹全文, 输出=判定)。"""
    per_trace = []
    for t in labeled_traces(traces):
        chars = sum(len(e.text or "") + len(e.reasoning or "") for e in t.events)
        toks_in = chars / CHARS_PER_TOKEN
        per_trace.append(toks_in / 1e6 * JUDGE_PRICE_IN
                         + JUDGE_OUT_TOKENS / 1e6 * JUDGE_PRICE_OUT)
    ours = 0.0
    judge_total = sum(per_trace)
    ratio = (ours / judge_total) if judge_total else 0.0
    return {"gate": "G4 成本", "threshold_ratio": G4_RATIO_MAX,
            "passed": ratio <= G4_RATIO_MAX,
            "ours_usd": ours, "ours_note": "纯 CPU 确定性管线, 零 LLM API 调用",
            "judge_est_per_trace_usd": round(mean(per_trace), 5) if per_trace else 0.0,
            "judge_est_200_usd": round(judge_total, 2),
            "ratio": round(ratio, 4),
            "assumptions": {
                "price_in_per_m": JUDGE_PRICE_IN, "price_out_per_m": JUDGE_PRICE_OUT,
                "out_tokens": JUDGE_OUT_TOKENS, "chars_per_token": CHARS_PER_TOKEN,
                "note": "judge 输入按轨迹全文估算 (json 非冗余文本), 输出按判定+理由 300 token"}}


def export_dataset(traces, attrs: Dict, out_path: str) -> Dict:
    """带失败归因标注的真实评测数据集 (resolved 人工标注 + 框架归因码)。"""
    rows = []
    for t in sorted(traces, key=lambda x: (x.model_key, x.trace_id)):
        a = attrs[(t.model_key, t.trace_id)]
        bm = t.meta.get("benchmark")
        rows.append({
            "model_key": t.model_key, "trace_id": t.trace_id,
            "task_id": t.task_id, "harness": t.harness,
            "benchmark": bm.get("name") if isinstance(bm, dict) else bm,
            "resolved": bm.get("resolved") if isinstance(bm, dict) else None,
            "reward": t.reward,
            "attribution_code": a["code"], "confidence": a.get("confidence"),
            "evidence": a.get("evidence", []),
            "trajectory_available": t.trajectory_available,
            "n_events": len(t.events), "cost_usd": t.cost_usd,
            "agent_seconds": t.agent_seconds,
        })
    doc = {"dataset": "agent-trace-attribution-v1", "generated_at":
           datetime.now().isoformat(timespec="seconds"), "n": len(rows),
           "annotations": ("resolved=人工/benchmark 标注; attribution_code=零 LLM "
                           "确定性归因 (码表见 ATTR_LABEL); evidence 可回溯到事件 seq"),
           "traces": rows}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    return doc


def run_gates(traces, attrs: Dict) -> Dict:
    g1 = gate1_consistency(traces, attrs)
    g2 = gate2_latency(traces)
    g3 = gate3_throughput(traces)
    g4 = gate4_cost(traces)
    return {"generated_at": datetime.now().isoformat(timespec="seconds"),
            "n_total": len(traces),
            "n_labeled": len(labeled_traces(traces)),
            "gates": [g1, g2, g3, g4],
            "all_passed": all(g["passed"] for g in (g1, g2, g3, g4))}
