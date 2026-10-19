# -*- coding: utf-8 -*-
"""失败轨迹语义序列层次聚类 (零依赖, 确定性)。

思路: 把失败轨迹的语义标签序列 (trace_semantic_seq, 已合并相邻重复)
看作短"词", 归一 Levenshtein 距离度量形态相似度, 平均 linkage 凝聚
聚类在固定阈值上切簇。每簇输出成员、归因码分布与代表轨迹 (medoid),
用于回答"失败长什么样、有几种形态"的概览问题。

产出为分析线索而非绝对结论: 阈值与标签粒度决定簇的粗细。
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

from .semantic import trace_semantic_seq

DEFAULT_THRESHOLD = 0.65   # 平均 linkage 距离超过即停止合并
DEFAULT_MIN_CALLS = 3      # 少于 3 个调用的轨迹不参与 (信息量不足)


def norm_levenshtein(a: Sequence[str], b: Sequence[str]) -> float:
    """归一编辑距离: levenshtein / max(len)。空对空=0, 空对非空=1。"""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    if la == 0 or lb == 0:
        return 1.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb] / max(la, lb)


def hierarchical_cluster(sequences: List[Sequence[str]],
                         threshold: float = DEFAULT_THRESHOLD) -> List[List[int]]:
    """平均 linkage 凝聚聚类, 返回索引簇列表 (每簇内部索引升序)。

    确定性: 每轮取全局最小簇间距离, 距离并列时按 (簇首索引) 字典序
    挑选最小者; 合并结果放入较低索引位置。
    """
    n = len(sequences)
    if n == 0:
        return []
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = norm_levenshtein(sequences[i], sequences[j])
            dist[i][j] = dist[j][i] = d
    clusters = [[i] for i in range(n)]
    while len(clusters) > 1:
        best = None  # (dist, a首成员, b首成员, a_idx, b_idx)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                s = 0.0
                for i in clusters[a]:
                    row = dist[i]
                    for j in clusters[b]:
                        s += row[j]
                d = s / (len(clusters[a]) * len(clusters[b]))
                key = (d, clusters[a][0], clusters[b][0])
                if best is None or key < best[:3]:
                    best = key + (a, b)
        d, _, _, a, b = best
        if d > threshold:
            break
        clusters[a] = sorted(clusters[a] + clusters[b])
        del clusters[b]
    return clusters


def cluster_failures(traces, attrs: Dict[Tuple[str, str], Dict],
                     threshold: float = DEFAULT_THRESHOLD,
                     min_calls: int = DEFAULT_MIN_CALLS) -> Dict:
    """失败轨迹聚类。attrs 为 {(model_key, trace_id): attribution dict}。

    失败判定: 归因码 != "success" (不可用轨迹/无调用轨迹被 min_calls 过滤)。
    """
    items = []  # (trace, code, seq)
    for t in traces:
        at = attrs.get((t.model_key, t.trace_id)) or {}
        if at.get("code") == "success":
            continue
        seq = trace_semantic_seq(t)
        if len(seq) < min_calls:
            continue
        items.append((t, at.get("code", ""), seq))
    groups = hierarchical_cluster([s for _, _, s in items], threshold)
    out = []
    for grp in groups:
        members = [items[i] for i in grp]
        medoid = min(members, key=lambda m: sum(
            norm_levenshtein(m[2], o[2]) for o in members))

        def _mt(m):
            return {"model_key": m[0].model_key, "trace_id": m[0].trace_id,
                    "task_id": m[0].task_id, "attribution": m[1],
                    "reward": m[0].reward, "n_calls": len(m[2])}
        out.append({
            "size": len(members),
            "attribution": dict(Counter(m[1] for m in members)),
            "medoid": _mt(medoid),
            "medoid_seq": medoid[2],
            "members": sorted((_mt(m) for m in members),
                              key=lambda d: (d["model_key"], d["trace_id"])),
        })
    out.sort(key=lambda c: (-c["size"], c["medoid"]["model_key"],
                            c["medoid"]["trace_id"]))
    return {"threshold": threshold, "min_calls": min_calls,
            "n_failed": len(items), "n_clusters": len(out), "clusters": out}
