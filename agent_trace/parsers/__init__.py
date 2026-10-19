# -*- coding: utf-8 -*-
"""解析器注册表。

新格式接入: 在子模块里定义 Parser 子类并 @register("harness名"),
parse_file 会按原始 JSON 的路由特征自动分发。
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, Optional

from ..schema import UnifiedTrace

_REGISTRY: Dict[str, "BaseParser"] = {}
_ROUTES: list = []  # [(detector_fn, parser_name)] 按注册顺序尝试


def register(harness: str, detector: Optional[Callable[[dict], bool]] = None):
    """@register("Claude Code") 注册解析器; detector 为无 harness 字段时的路由探测。"""

    def deco(cls):
        inst = cls()
        _REGISTRY[harness.lower()] = inst
        if detector is not None:
            _ROUTES.append((detector, harness.lower()))
        return cls

    return deco


def get_parser(harness: str):
    return _REGISTRY.get((harness or "").strip().lower())


def parse_file(path: str, model_key: str = "") -> UnifiedTrace:
    """解析单个轨迹文件 → UnifiedTrace。按文件内容自动路由到对应解析器。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    parser = _pick_parser(raw)
    return parser.parse(raw, path=path, model_key=model_key)


def _pick_parser(raw: dict):
    harness = raw.get("harness")
    if harness and harness.strip().lower() in _REGISTRY:
        return _REGISTRY[harness.strip().lower()]
    # 无 harness 字段: 按探测函数路由 (先注册先试)
    for detector, name in _ROUTES:
        try:
            if detector(raw):
                return _REGISTRY[name]
        except Exception:
            continue
    # ALE-CLI 信封兜底 (未知 harness 名但共享同一消息结构)
    if "benchmark" in raw or "messages" in raw:
        return _REGISTRY.get("__ale_generic__")
    raise ValueError("无法识别的轨迹格式: keys=%s" % sorted(raw.keys())[:12])


class BaseParser:
    """解析器基类: 子类实现 parse(raw, path, model_key)。"""

    name = "base"

    def parse(self, raw: dict, path: str = "", model_key: str = "") -> UnifiedTrace:
        raise NotImplementedError


# 导入子模块触发注册
from . import alecli  # noqa: E402,F401  (注册 Claude Code / DSH / ALE 兜底)
from . import miniswe  # noqa: E402,F401  (注册 minisweagent, SWE-bench 嵌套布局)
