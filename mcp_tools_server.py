"""
Ubuntu MCP Tools Server

This file implements a practical set of **Model Context Protocol** (MCP) tools for
an LLM to use on Ubuntu Linux. It uses Anthropic's `fastmcp` Python helper to expose
well-typed tools over MCP.

Dependencies (Ubuntu):
  sudo apt-get update && sudo apt-get install -y strace smartmontools

Python deps:
  pip install fastmcp psutil pydantic

Run (stdio):
  python mcp_tools_server.py

Notes:
- Some tools require elevated privileges (e.g., strace/eBPF, SMART). Use sudo as needed.
- Each tool validates inputs and returns structured JSON-friendly data.
- Tools are designed to be non-destructive; any mutating ops are intentionally omitted.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import socket
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import psutil
from pydantic import BaseModel, Field

try:
    from fastmcp import FastMCP
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "fastmcp is required. Install with: pip install fastmcp"
    ) from e


# -----------------------------
# Utility helpers
# -----------------------------

def _which(cmd: str) -> Optional[str]:
    """Return full path to cmd if found in PATH, else None."""
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(p) / cmd
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def _require_tool(cmd: str) -> None:
    if _which(cmd) is None:
        raise RuntimeError(f"Required tool '{cmd}' not found in PATH. Please install it.")


def _run(cmd: List[str], timeout: int = 60) -> Tuple[int, str, str]:
    """Run a command and return (rc, stdout, stderr) as text."""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


# -----------------------------
# Pydantic models (for nice schemas)
# -----------------------------

class TopProcess(BaseModel):
    pid: int
    name: str
    cpu_percent: float
    rss_mb: float = Field(description="Resident Set Size in MB")
    username: Optional[str] = None
    exe: Optional[str] = None


class Listener(BaseModel):
    pid: Optional[int]
    laddr: str
    raddr: Optional[str]
    status: Optional[str]
    exe: Optional[str]
    user: Optional[str]
    fd: Optional[int]


class Kmod(BaseModel):
    name: str
    size: int
    used_by: List[str]


class KmodInfo(BaseModel):
    name: str
    fields: Dict[str, str]


class SmartAttr(BaseModel):
    device: str
    health: Optional[str]
    attrs: Dict[str, str]


class SyscallSummary(BaseModel):
    duration_s: int
    mode: Literal["pid", "command"]
    target: str
    summary_table: List[Dict[str, str]] = Field(
        description="Parsed strace -c table by syscall"
    )
    stderr_tail: Optional[str] = None


class LargeFile(BaseModel):
    path: str
    size_mb: float


class DNSCheck(BaseModel):
    domain: str
    addrs: List[str]
    elapsed_ms: float


# -----------------------------
# MCP server
# -----------------------------

app = FastMCP("ubuntu-tools")


# 1) Processes / Top
# --- add this helper above the @app.tool() block ---
def get_top_impl(by: Literal["cpu","mem"], limit: int = 10) -> List[TopProcess]:
    limit = max(1, min(limit, 100))

    # Prime CPU measurement window (two passes; psutil needs a delay)
    for p in psutil.process_iter(attrs=["pid"]):
        try:
            p.cpu_percent(None)
        except Exception:
            pass
    time.sleep(1.0)  # was 0.2; longer = more reliable on some systems

    procs: List[TopProcess] = []
    for p in psutil.process_iter(attrs=["pid","name","username","memory_info","exe"]):
        try:
            cpu = p.cpu_percent(None)
            rss_mb = (p.memory_info().rss or 0) / (1024*1024)
            procs.append(
                TopProcess(
                    pid=p.pid,
                    name=p.info.get("name") or str(p.pid),
                    cpu_percent=cpu,
                    rss_mb=rss_mb,
                    username=p.info.get("username"),
                    exe=p.info.get("exe"),
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if by == "cpu":
        procs.sort(key=lambda x: x.cpu_percent, reverse=True)
    else:
        procs.sort(key=lambda x: x.rss_mb, reverse=True)
    return procs[:limit]

# --- keep the MCP tool, but delegate to the helper ---
@app.tool()
def get_top(by: Literal["cpu", "mem"], limit: int = 10) -> List[TopProcess]:
    return get_top_impl(by=by, limit=limit)


# 2) Network listeners
@app.tool()
def list_listeners(ports: Optional[List[int]] = None) -> List[Listener]:
    """List TCP/UDP internet listeners and connections. Optionally filter by port(s)."""
    conns = psutil.net_connections(kind="inet")
    out: List[Listener] = []
    for c in conns:
        try:
            laddr = f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else ""
            raddr = f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else None
            if ports and c.laddr and c.laddr.port not in set(ports):
                continue
            exe = None
            user = None
            if c.pid:
                try:
                    p = psutil.Process(c.pid)
                    exe = p.exe() if p else None
                    user = p.username() if p else None
                except Exception:
                    pass
            out.append(
                Listener(
                    pid=c.pid,
                    laddr=laddr,
                    raddr=raddr,
                    status=getattr(c, "status", None),
                    exe=exe,
                    user=user,
                    fd=getattr(c, "fd", None),
                )
            )
        except Exception:
            continue
    return out


# 3) Kernel modules
@app.tool()
def list_kmods() -> List[Kmod]:
    """Return loaded kernel modules (lsmod)."""
    rc, out, err = _run(["lsmod"])  # present on Ubuntu by default
    if rc != 0:
        raise RuntimeError(err or "lsmod failed")

    lines = out.strip().splitlines()
    mods: List[Kmod] = []
    for line in lines[1:]:  # skip header
        parts = line.split()
        if len(parts) >= 3:
            name, size, used = parts[0], int(parts[1]), parts[2]
            used_by: List[str] = []
            if len(parts) >= 4:
                used_by = parts[3].split(",") if parts[3] != "-" else []
            mods.append(Kmod(name=name, size=size, used_by=used_by))
    return mods


@app.tool()
def kmod_info(name: str) -> KmodInfo:
    """Return `modinfo` fields for a kernel module by name."""
    _require_tool("modinfo")
    rc, out, err = _run(["modinfo", name])
    if rc != 0:
        raise RuntimeError(err or f"modinfo failed for {name}")
    fields: Dict[str, str] = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    return KmodInfo(name=name, fields=fields)


# 4) Disk SMART (health)
@app.tool()
def disk_smart(devices: Optional[List[str]] = None) -> List[SmartAttr]:
    """Query SMART health for block devices (via smartctl). Requires smartmontools.

    On most Ubuntu systems, try devices like: /dev/sda, /dev/nvme0n1
    """
    _require_tool("smartctl")

    # If no devices provided, try to discover basic list from /sys/block
    if not devices:
        devs = []
        for p in Path("/sys/block").iterdir():
            name = p.name
            if name.startswith("loop"):
                continue
            # NVMe live under /dev/nvmeXnY, others under /dev/<name>
            if name.startswith("nvme"):
                devs.append(f"/dev/{name}")
            else:
                devs.append(f"/dev/{name}")
        devices = devs

    results: List[SmartAttr] = []
    for dev in devices:
        rc, out, err = _run(["smartctl", "-H", "-A", dev], timeout=20)
        if rc != 0:
            results.append(SmartAttr(device=dev, health=None, attrs={"error": err or out}))
            continue
        health = None
        attrs: Dict[str, str] = {}
        for line in out.splitlines():
            if "SMART overall-health self-assessment test result" in line or "SMART Health Status" in line:
                health = line.split(":")[-1].strip()
            elif re.match(r"^\s*\d+\s+\S+", line):
                # Attribute table row; keep raw_value at the end
                parts = line.split()
                if len(parts) >= 10:
                    attr_name = parts[1]
                    raw_val = parts[-1]
                    attrs[attr_name] = raw_val
        results.append(SmartAttr(device=dev, health=health, attrs=attrs))
    return results


# 5) Find large files
@app.tool()
def find_large(path: str, min_mb: int = 100, limit: int = 50) -> List[LargeFile]:
    """Scan `path` recursively and return files >= min_mb, largest first."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

    matches: List[Tuple[float, Path]] = []
    min_bytes = min_mb * 1024 * 1024
    for root, dirs, files in os.walk(p, followlinks=False):
        # Skip special/system dirs that can be huge/slow
        if any(seg in {"/proc", "/sys", "/dev"} for seg in [root]):
            continue
        for f in files:
            try:
                fp = Path(root) / f
                st = fp.stat()
                if st.st_size >= min_bytes:
                    matches.append((st.st_size / (1024 * 1024), fp))
            except (FileNotFoundError, PermissionError):
                continue
    matches.sort(key=lambda x: x[0], reverse=True)
    return [LargeFile(path=str(fp), size_mb=size) for size, fp in matches[:limit]]


# 6) DNS healthcheck
@app.tool()
def dns_healthcheck(domains: List[str]) -> List[DNSCheck]:
    """Resolve domains and measure latency."""
    results: List[DNSCheck] = []
    for d in domains:
        t0 = time.perf_counter()
        addrs: List[str] = []
        try:
            infos = socket.getaddrinfo(d, None)
            for fam, stype, proto, canon, sockaddr in infos:
                host = sockaddr[0] if isinstance(sockaddr, tuple) else str(sockaddr)
                if host not in addrs:
                    addrs.append(host)
            elapsed = (time.perf_counter() - t0) * 1000.0
            results.append(DNSCheck(domain=d, addrs=addrs, elapsed_ms=elapsed))
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            results.append(DNSCheck(domain=d, addrs=[f"error: {e}"], elapsed_ms=elapsed))
    return results


# 7) Syscall sampling via strace (-c)
@app.tool()
def trace_syscalls(
    pid: Optional[int] = None,
    command: Optional[str] = None,
    duration_s: int = 10,
) -> SyscallSummary:
    """Summarize syscalls for a PID or a new command over a short window using `strace -c`.

    Notes:
      - Requires `strace` and sufficient privileges. Use with care.
      - Only one of `pid` or `command` must be provided.
    """
    _require_tool("strace")
    if (pid is None) == (command is None):
        raise ValueError("Provide exactly one of 'pid' or 'command'.")

    with tempfile.TemporaryDirectory() as td:
        out_file = Path(td) / "strace.sum"
        if pid is not None:
            cmd = [
                "sudo",
                "strace",
                "-f",
                "-qq",
                "-c",
                "-p",
                str(pid),
            ]
        else:
            # Run a new command under strace
            cmd = ["sudo", "strace", "-f", "-qq", "-c"] + shlex.split(command)  # type: ignore

        # Use timeout to stop after duration_s
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                proc.wait(timeout=duration_s)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception as e:
            raise RuntimeError(f"strace launch failed: {e}")

        # strace -c prints a summary table to stderr
        stderr = proc.stderr.read() if proc.stderr else ""
        # Parse the classic -c table
        table: List[Dict[str, str]] = []
        parsing = False
        for line in stderr.splitlines():
            if re.match(r"\s*%\s*time\s+seconds\s+usecs/call\s+calls\s+errors\s+syscall", line):
                parsing = True
                continue
            if parsing:
                if line.strip().startswith("------"):
                    continue
                if line.strip().startswith("% time"):
                    continue
                if line.strip().startswith("total"):
                    # End of table
                    break
                parts = line.strip().split()
                if len(parts) >= 6:
                    # Join the syscall name (rest of the line) because it may include spaces on exotic systems
                    syscall = parts[5]
                    row = {
                        "% time": parts[0],
                        "seconds": parts[1],
                        "usecs/call": parts[2],
                        "calls": parts[3],
                        "errors": parts[4],
                        "syscall": syscall,
                    }
                    table.append(row)
        tail = "\n".join(stderr.splitlines()[-20:]) if stderr else None

    mode = "pid" if pid is not None else "command"
    target = str(pid) if pid is not None else command or ""
    return SyscallSummary(duration_s=duration_s, mode=mode, target=target, summary_table=table, stderr_tail=tail)


if __name__ == "__main__":
    # Run the MCP server over stdio
    app.run()
