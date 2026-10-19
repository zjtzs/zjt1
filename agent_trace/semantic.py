# -*- coding: utf-8 -*-
"""语义归一: 把异构工具名/命令映射到 10 个语义标签。

标签集: EXEC READ SEARCH EDIT WRITE PLAN GUI JOB COMM OTHER
- 精确映射表命中 → conf=1.0
- shell 命令启发式 (exec 系 harness 无专用工具, 全靠 bash 写文件) → conf=0.85
- 兜底 OTHER → conf=0.5
观测 (tool_result) 侧标签统一 conf=0.9 (standard.py 约定)。
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# 10 个语义标签
EXEC = "EXEC"      # 执行命令/运行代码
READ = "READ"      # 读文件/读资源
SEARCH = "SEARCH"  # 检索 (grep/glob/web search)
EDIT = "EDIT"      # 编辑文件
WRITE = "WRITE"    # 整文件写入
PLAN = "PLAN"      # 计划/任务管理
GUI = "GUI"        # 浏览器/截图操作
JOB = "JOB"        # 子 agent / 后台任务
COMM = "COMM"      # 通讯 (MCP 资源列表/webfetch/用户交互)
OTHER = "OTHER"

TOOL_TAGS: Dict[str, str] = {
    # --- Claude Code 系 (PascalCase) ---
    "bash": EXEC, "read": READ, "write": WRITE, "edit": EDIT,
    "multiedit": EDIT, "glob": SEARCH, "grep": SEARCH, "ls": SEARCH,
    "task": JOB, "todowrite": PLAN, "todoread": PLAN,
    "exitplanmode": PLAN, "enterplanmode": PLAN,
    "webfetch": COMM, "websearch": SEARCH,
    "notebookedit": EDIT, "notebookread": READ,
    "list_mcp_resources": COMM, "listmcpresources": COMM,
    "readmcpresource": COMM, "listmcpresourcetemplates": COMM,
    "computer": GUI, "screenshot": GUI, "browser_click": GUI,
    "browser_navigate": GUI, "browser_snapshot": GUI,
    "sendmessage": COMM, "askuserquestion": COMM,
    "requestauthorization": COMM, "showwidget": COMM,
    # --- job / goal 系 (DSH subagent 控制与目标管理) ---
    "job_kill": JOB, "job_list": JOB, "job_output": JOB, "skill": JOB,
    "update_goal": PLAN, "get_goal": PLAN,
    # --- exec 系 (codex/gpt) ---
    "exec_command": EXEC, "execcommand": EXEC, "shell": EXEC,
    "shell_command": EXEC, "run_command": EXEC, "terminal": EXEC,
    "apply_patch": EDIT, "applypatch": EDIT, "update_plan": PLAN,
    "view_image": READ, "read_file": READ, "write_file": WRITE,
    "search_codebase": SEARCH, "searchcodenase": SEARCH,
    "stop_command": EXEC, "delegatetool": JOB, "new_context": PLAN,
}

# mcp__ 前缀的第三方工具
_MCP_RE = re.compile(r"^mcp__")


def tag_for_tool(raw_name: str) -> Tuple[str, float]:
    """工具名 → (语义标签, 置信度)。大小写不敏感。"""
    key = (raw_name or "").strip().lower()
    if not key:
        return OTHER, 0.5
    if key in TOOL_TAGS:
        return TOOL_TAGS[key], 1.0
    if _MCP_RE.match(key):
        # 计算机使用类 MCP (mcp__cua__*) 是 GUI 操作
        return (GUI, 0.9) if key.startswith("mcp__cua__") else (COMM, 0.9)
    # 未收录: 按词面启发式兜底
    if "read" in key or "view" in key or "open" in key:
        return READ, 0.8
    if "search" in key or "find" in key or "grep" in key or "glob" in key:
        return SEARCH, 0.8
    if "edit" in key or "patch" in key:
        return EDIT, 0.8
    if "write" in key or "create" in key:
        return WRITE, 0.8
    if "plan" in key or "todo" in key:
        return PLAN, 0.8
    if "exec" in key or "run" in key or "command" in key:
        return EXEC, 0.8
    return OTHER, 0.5


# ---------------- shell 命令级语义 (exec 系 harness 的文件写入检测) ----------------

# 写文件形态: heredoc / tee / 重定向 / touch / mkdir / cp / mv。
# 注意 2>/dev/null 只是把 stderr 静默, 不是状态改变, 必须负向断言排除。
_RE_HEREDOC = re.compile(r"<<\s*-?\s*['\"]?(\w+)")
_RE_TEE = re.compile(r"(?:^|[|;&]\s*)tee(?:\s|-)")
_RE_REDIRECT = re.compile(r"(?<![0-9><])>{1,2}\s*[^&\s]")
_RE_MKFS = re.compile(r"(?:^|[|;&]\s*)(?:touch|mkdir|cp|mv|install)\s+\S")

_RE_POLLING = re.compile(r"(?:^|[|;&]\s*)(?:sleep|tail|watch|write_stdin|read)\b")


def classify_shell_command(cmd: str) -> Tuple[str, float]:
    """bash 命令 → (语义标签, 置信度)。供 exec 系轨迹的调用与产物检测使用。"""
    c = (cmd or "").strip()
    if not c:
        return OTHER, 0.5
    # 写文件: heredoc/tee/重定向/touch 系。2>/dev/null 单独出现不算。
    probe = re.sub(r"\d?>\s*/dev/null", "", c)
    if (_RE_HEREDOC.search(probe) or _RE_TEE.search(probe)
            or _RE_REDIRECT.search(probe) or _RE_MKFS.search(probe)):
        # cat file > other 之类读为主的重定向也算产物步 (文件确实变了)
        return WRITE, 0.85
    if _RE_POLLING.search(c):
        return READ, 0.85
    first = re.split(r"[|;&\s]", c, 1)[0].lower()
    if first in ("cat", "head", "less", "more", "find", "grep", "rg", "ls",
                 "wc", "which", "file", "stat", "du", "df", "env", "printenv"):
        return READ if first not in ("find", "grep", "rg", "ls") else SEARCH, 0.85
    if first in ("cd", "pwd", "echo", "export", "set", "source", "true", "false"):
        return EXEC, 0.85
    if first in ("python", "python3", "pytest", "pip", "node", "npm", "cargo",
                 "go", "make", "gcc", "g++", "java", "time"):
        return EXEC, 0.85
    return EXEC, 0.85


def trace_semantic_seq(trace, collapse: bool = True):
    """轨迹 → 语义标签序列 (按 assistant tool_calls 顺序)。

    collapse=True 时合并相邻重复标签 (READ READ READ → READ), 降低长循环
    对序列距离/n-gram 支持度的支配。供 clustering / pathmining 使用。
    """
    seq = []
    for e in trace.events:
        if e.kind != "assistant":
            continue
        for c in e.tool_calls:
            tag, _ = semantic_for_call(c.name, c.args)
            if collapse and seq and seq[-1] == tag:
                continue
            seq.append(tag)
    return seq


def semantic_for_call(raw_name: str, args: Optional[Dict] = None) -> Tuple[str, float]:
    """调用侧语义: 工具名优先; exec 系工具再细看命令内容 (写文件→WRITE)。"""
    tag, conf = tag_for_tool(raw_name)
    if raw_name and raw_name.strip().lower() in ("exec_command", "execcommand", "shell",
                                                 "shell_command", "run_command", "terminal",
                                                 "bash") and args:
        cmd = args.get("command") or args.get("cmd") or args.get("script") or ""
        if cmd:
            stag, sconf = classify_shell_command(str(cmd))
            if stag == WRITE:  # 命令级写文件优先于工具级 EXEC
                return WRITE, max(conf, sconf)
    return tag, conf
