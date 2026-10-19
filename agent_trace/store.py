# -*- coding: utf-8 -*-
"""存储层: 扫描数据目录 + SQLite 增量缓存。

增量规则: 按 source_path (normcase, Windows 跨进程一致性) + mtime + size
失效判断; 未变更复用, 新/改重解析 upsert, 删除自动清理。
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Dict, List, Tuple

from .parsers import parse_file
from .schema import UnifiedTrace

# 项目根下非模型目录 (防把报告/代码目录当模型扫进来)
EXCLUDE_DIRS = {"cache", "datasets", "docs", "static", "tests", "schemas",
                "scripts", "agent_trace", "__pycache__", ".trae", "node_modules",
                "datasets_backup"}


def discover_model_dirs(data_dir: str) -> Dict[str, str]:
    """发现模型目录: {model_key: dir_path}。

    两种布局:
    - ALE 平铺: <model>/<file>.json
    - miniswe 嵌套: <model>/<instance_id>/<instance_id>.traj.json
    """
    models: Dict[str, str] = {}
    for name in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, name)
        if not os.path.isdir(d) or name in EXCLUDE_DIRS or name.startswith((".", "_")):
            continue
        entries = os.listdir(d)
        if any(e.endswith(".json") and os.path.isfile(os.path.join(d, e))
               for e in entries):
            models[name] = d  # ALE 平铺
            continue
        for e in entries:  # miniswe 嵌套
            sub = os.path.join(d, e)
            if os.path.isdir(sub) and any(f.endswith(".traj.json") for f in os.listdir(sub)):
                models[name] = d
                break
    return models


def discover_trace_files(model_dir: str) -> List[str]:
    """目录内的轨迹文件清单 (两种布局都支持)。"""
    files: List[str] = []
    for name in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p) and name.endswith(".json"):
            files.append(p)
        elif os.path.isdir(p) and not name.startswith("."):
            for f in sorted(os.listdir(p)):
                if f.endswith(".traj.json") and os.path.isfile(os.path.join(p, f)):
                    files.append(os.path.join(p, f))
    return files


def load_benchmark_labels(data_dir: str, model_key: str) -> dict:
    """加载 per_instance_details__{model_key}.json → {instance_id: {resolved, cost, api_calls}}。"""
    p = os.path.join(data_dir, f"per_instance_details__{model_key}.json")
    if not os.path.isfile(p):
        return {}
    try:
        import json
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if isinstance(v, dict)}
    except (ValueError, OSError):
        return {}


class TraceStore:
    def __init__(self, data_dir: str, cache_dir: str = "cache"):
        self.data_dir = os.path.abspath(data_dir)
        self.cache_dir = os.path.abspath(cache_dir)
        self.db_path = os.path.join(self.cache_dir, "traces.db")

    # ---------------- SQLite ----------------
    def _connect(self) -> sqlite3.Connection:
        os.makedirs(self.cache_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS traces ("
            " source_path TEXT PRIMARY KEY,"
            " model_key TEXT NOT NULL,"
            " mtime REAL, size INTEGER,"
            " payload TEXT NOT NULL)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_model ON traces(model_key)")
        return conn

    def build(self, force: bool = False) -> Tuple[List[UnifiedTrace], Dict]:
        """增量构建并返回全部轨迹。返回 (traces, stats)。"""
        t0 = time.perf_counter()
        models = discover_model_dirs(self.data_dir)
        want: Dict[str, Tuple[str, float, int]] = {}  # key -> (model_key, mtime, size)
        for model_key, mdir in models.items():
            for f in discover_trace_files(mdir):
                st = os.stat(f)
                key = os.path.normcase(os.path.abspath(f))  # Windows 跨进程一致性
                want[key] = (model_key, st.st_mtime, st.st_size)

        conn = self._connect()
        try:
            rows = {r[0]: (r[1], r[2], r[3], r[4])
                    for r in conn.execute("SELECT source_path, model_key, mtime, size, payload FROM traces")}
            traces: List[UnifiedTrace] = []
            reused = reparsed = 0
            for key, (model_key, mtime, size) in want.items():
                hit = rows.get(key)
                if not force and hit and hit[1] == mtime and hit[2] == size and hit[0] == model_key:
                    try:
                        traces.append(UnifiedTrace.from_dict(__import__("json").loads(hit[3])))
                        reused += 1
                        continue
                    except (ValueError, KeyError, TypeError):
                        pass  # 反序列化失败 → 重解析
                t = parse_file(key, model_key=model_key)
                conn.execute("INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?)",
                             (key, model_key, mtime, size, t.to_json()))
                traces.append(t)
                reparsed += 1
            cleaned = 0
            for key in rows:
                if key not in want:
                    conn.execute("DELETE FROM traces WHERE source_path=?", (key,))
                    cleaned += 1
            conn.commit()
            # benchmark 标注注入 (per_instance_details → meta.benchmark), 不参与缓存失效
            labels_by_model = {mk: load_benchmark_labels(self.data_dir, mk)
                               for mk in models}
            labeled = 0
            for t in traces:
                labels = labels_by_model.get(t.model_key) or {}
                rec = labels.get(t.task_id)
                if rec is not None:
                    t.meta["benchmark"] = {
                        "name": "swe-bench",
                        "resolved": rec.get("resolved"),
                        "cost": rec.get("cost"),
                        "api_calls": rec.get("api_calls"),
                    }
                    labeled += 1
        finally:
            conn.close()
        traces.sort(key=lambda t: (t.model_key, t.trace_id))
        stats = {"total": len(traces), "reused": reused, "reparsed": reparsed,
                 "cleaned": cleaned, "models": sorted(models.keys()),
                 "labeled": labeled,
                 "elapsed": round(time.perf_counter() - t0, 3)}
        return traces, stats
