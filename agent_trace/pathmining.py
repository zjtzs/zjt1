# -*- coding: utf-8 -*-
"""语义 n-gram 频繁序列挖掘: 成功 vs 失败组对照 (零依赖, 确定性)。

按 resolved 标注 (meta.benchmark.resolved) 分组, 缺标注时用 reward 兜底
(>=1 成功)。挖掘组内轨迹中出现过的连续语义 n-gram (n=2..4), 支持度 =
含该 n-gram 的轨迹比例 (文档级, 一条轨迹重复出现只计一次)。

对照输出 fail_enriched / success_enriched: 对方组几乎不出现而本组高频
的模式, 用于回答"哪些行为序列与失败/成功相关"。相关性线索, 非因果。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .semantic import trace_semantic_seq

N_MIN, N_MAX = 2, 4
TOP_K = 15
MIN_SUPPORT = 0.05   # 组内支持度门槛


def group_of(trace) -> Optional[str]:
    """resolved 优先, reward 兜底; 两者皆无 → None (不参与分组)。"""
    bm = trace.meta.get("benchmark")
    if isinstance(bm, dict):
        r = bm.get("resolved")
        if r is True:
            return "success"
        if r is False:
            return "fail"
    if trace.reward is not None:
        return "success" if trace.reward >= 1.0 else "fail"
    return None


def _grams(seq: List[str], n_min: int, n_max: int) -> set:
    out = set()
    for n in range(n_min, n_max + 1):
        for i in range(len(seq) - n + 1):
            out.add(tuple(seq[i:i + n]))
    return out


def mine_ngrams(traces, n_min: int = N_MIN, n_max: int = N_MAX,
                top_k: int = TOP_K, min_support: float = MIN_SUPPORT) -> Dict:
    groups: Dict[str, list] = {"success": [], "fail": []}
    for t in traces:
        g = group_of(t)
        if g and t.trajectory_available:
            groups[g].append(t)
    full: Dict[str, Dict[tuple, float]] = {}
    out = {"n_traces": {g: len(ts) for g, ts in groups.items()}, "groups": {}}
    for g, ts in groups.items():
        counter: Dict[tuple, int] = {}
        for t in ts:
            for gram in _grams(trace_semantic_seq(t), n_min, n_max):
                counter[gram] = counter.get(gram, 0) + 1
        denom = max(len(ts), 1)
        full[g] = {k: v / denom for k, v in counter.items()}
        pats = [{"ngram": list(k), "n": len(k), "n_traces": v, "support": v / denom}
                for k, v in counter.items() if v / denom >= min_support]
        # 确定性排序: 支持度降序 → 长度降序 → 字典序
        pats.sort(key=lambda p: (-p["support"], -p["n"], p["ngram"]))
        out["groups"][g] = {"patterns": pats[:top_k]}
    # 对照: 对方组支持度越低、本组越高越典型; 对方为 0 记 lift=None (纯判别)
    for g, other in (("fail", "success"), ("success", "fail")):
        rows = []
        for k, s_self in full[g].items():
            if s_self < min_support:
                continue
            s_other = full[other].get(k, 0.0)
            rows.append({"ngram": list(k), "n": len(k),
                         "support_self": s_self, "support_other": s_other,
                         "lift": (s_self / s_other) if s_other > 0 else None})
        # lift=None (对方零出现) 排最前, 其余按 lift 降序; 并列按支持度/字典序
        rows.sort(key=lambda r: (r["lift"] is not None, -(r["lift"] or 0),
                                 -r["support_self"], r["ngram"]))
        out["groups"][g]["enriched"] = rows[:top_k]
    return out
