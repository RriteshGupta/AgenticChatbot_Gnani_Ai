import asyncio
import shutil
import subprocess
import sys
import threading
import traceback
from pathlib import Path

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER = {
    "calculator": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(Path(__file__).parent / "server.py")],
    },
    "manim-server": {
        "transport": "stdio",
        "command": "C:\\Program Files\\Python313\\python.exe",
        "args": ["C:\\Users\\rites\\Desktop\\manim-mcp-server\\src\\manim_server.py"],
        "env": {
            "MANIM_EXECUTABLE": "C:\\Users\\rites\\AppData\\Roaming\\Python\\Python313\\Scripts\\manim.exe"
        },
    },
}

# Har server ke Python mein ye modules install hone chahiye
REQUIRES = {
    "calculator": ["mcp"],
    "manim-server": ["mcp", "manim"],
}

TIMEOUT = 180  # sec, manim server dheere start hota hai

# Frontend chahe to ye dekh sakta hai: {"calculator": "ok", "manim-server": "fail: ..."}
MCP_STATUS = {}


def _log(msg):
    print(f"[MCP] {msg}", flush=True)


def _run(coro):
    """asyncio.run, lekin agar pehle se loop chal raha ho to alag thread mein chalata hai."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box = {}

    def target():
        try:
            box["result"] = asyncio.run(coro)
        except BaseException as e:
            box["error"] = e

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


def _flatten(e):
    """ExceptionGroup ke andar chhupi asli errors nikalta hai."""
    if isinstance(e, BaseExceptionGroup):
        for sub in e.exceptions:
            yield from _flatten(sub)
    else:
        yield e


def _precheck(name, cfg):
    """Server chalane se pehle aam galtiyan pakadta hai. Problems ki list lautata hai."""
    problems = []

    cmd = cfg.get("command", "")
    cmd_ok = Path(cmd).exists() if ("\\" in cmd or "/" in cmd) else shutil.which(cmd)
    if not cmd_ok:
        problems.append(f"command nahi mila: {cmd}")

    for arg in cfg.get("args", []):
        if str(arg).lower().endswith(".py") and not Path(arg).exists():
            problems.append(f"script file nahi mili: {arg}")

    for key, val in cfg.get("env", {}).items():
        if ("\\" in val or "/" in val) and not Path(val).exists():
            problems.append(f"env {key} ka path nahi mila: {val}")

    # Python modules us Python mein install hain ya nahi
    if cmd_ok and "python" in Path(cmd).name.lower():
        for mod in REQUIRES.get(name, []):
            try:
                r = subprocess.run(
                    [cmd, "-c", f"import {mod}"],
                    capture_output=True, text=True, timeout=60,
                )
                if r.returncode != 0:
                    last = (r.stderr.strip().splitlines() or ["?"])[-1]
                    problems.append(f"'{mod}' us Python mein import nahi hua ({last})")
            except Exception as e:
                problems.append(f"'{mod}' check fail: {e}")

    return problems


def _to_sync_tool(mcp_tool):
    """Async MCP tool ko sync tool bana deta hai, taaki ToolNode usse chala sake."""
    def _call(**kwargs):
        try:
            return _run(mcp_tool.ainvoke(kwargs))
        except Exception as e:
            subs = "; ".join(f"{type(s).__name__}: {s}" for s in _flatten(e))
            _log(f"tool '{mcp_tool.name}' chalte waqt error: {subs}")
            return f"Tool '{mcp_tool.name}' fail hua: {subs}"

    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema=mcp_tool.args_schema,
        func=_call,
    )


def load_mcp_tools():
    """Har server alag load hota hai. Ek fail ho to baaki chalte rehte hain."""
    all_tools = []

    for name, cfg in SERVER.items():
        _log(f"--- {name} ---")

        problems = _precheck(name, cfg)
        if problems:
            for p in problems:
                _log(f"  ❌ {p}")
            MCP_STATUS[name] = "fail: " + " | ".join(problems)
            continue
        _log("  ✔ paths aur modules theek hain, server start kar raha hoon...")

        try:
            client = MultiServerMCPClient({name: cfg})
            tools = _run(asyncio.wait_for(client.get_tools(), timeout=TIMEOUT))
            all_tools += [_to_sync_tool(t) for t in tools]
            names = ", ".join(t.name for t in tools)
            _log(f"  ✅ {len(tools)} tools mile: {names}")
            MCP_STATUS[name] = "ok"
        except asyncio.TimeoutError:
            msg = f"{TIMEOUT} sec mein server ne jawab nahi diya (hang ya bahut slow)"
            _log(f"  ❌ {msg}")
            MCP_STATUS[name] = "fail: " + msg
        except Exception as e:
            subs = [f"{type(s).__name__}: {s}" for s in _flatten(e)]
            for s in subs:
                _log(f"  ❌ {s}")
            _log("  (poora traceback neeche)")
            traceback.print_exc()
            MCP_STATUS[name] = "fail: " + " | ".join(subs)

    _log("=== Summary ===")
    for name, status in MCP_STATUS.items():
        _log(f"{name}: {status}")
    return all_tools


if __name__ == "__main__":
    tools = load_mcp_tools()
    print(f"\nTotal tools: {len(tools)}")