import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

@pytest.fixture
def tools():
    # Replace only transport registration; execute the actual tool implementations.
    transport = SimpleNamespace(tool=lambda:lambda function:function)
    spec = importlib.util.spec_from_file_location('tools_under_test', Path(__file__).resolve().parents[1] / 'mcp_tools_server.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'fastmcp':SimpleNamespace(FastMCP=lambda name:transport), 'tools_under_test':module}):
        spec.loader.exec_module(module)
    return module

def test_kernel_module_output_is_parsed(tools, monkeypatch):
    monkeypatch.setattr(tools, '_run', lambda cmd:(0,'Module Size Used by\nkvm 100 1 kvm_intel\nempty 50 0 -\n',''))
    result = tools.list_kmods()
    assert [(item.name,item.size,item.used_by) for item in result] == [('kvm',100,['kvm_intel']),('empty',50,[])]

def test_failed_kernel_command_surfaces_error(tools, monkeypatch):
    monkeypatch.setattr(tools, '_run', lambda cmd:(1,'','permission denied'))
    with pytest.raises(RuntimeError, match='permission denied'):
        tools.list_kmods()

def test_top_processes_sort_by_memory_and_limit(tools, monkeypatch):
    def process(pid, rss):
        return SimpleNamespace(pid=pid,info={'name':str(pid)},cpu_percent=lambda _:0,memory_info=lambda:SimpleNamespace(rss=rss*1024*1024))
    monkeypatch.setattr(tools.psutil, 'process_iter', lambda **kw:[process(1,10),process(2,30)])
    monkeypatch.setattr(tools.time, 'sleep', lambda _:None)
    result = tools.get_top_impl('mem',1)
    assert len(result) == 1
    assert (result[0].pid,result[0].rss_mb) == (2,30)

@pytest.mark.parametrize('row,errors', [
    ('89.76 0.008016 4 1912 getdents', '0'),
    ('0.00 0.000000 0 3 3 access', '3'),
])
def test_trace_summary_preserves_calls_with_or_without_errors(tools, monkeypatch, row, errors):
    import io
    # Classic -c layout: https://strace.io/ (Count time, calls, and errors).
    # Fixture is synthetic; no live tracing or privileged process is launched.
    summary = ('% time seconds usecs/call calls errors syscall\n'
               '------ ----------- ----------- --------- --------- ----------------\n'
               + row + '\n'
               '100.00 0.008930 25440 3 total\n')
    process = MagicMock()
    process.stderr = io.StringIO(summary)
    monkeypatch.setattr(tools, '_require_tool', lambda name: None)
    monkeypatch.setattr(tools.subprocess, 'Popen', lambda *args, **kwargs: process)
    result = tools.trace_syscalls(pid=123, duration_s=1)
    assert len(result.summary_table) == 1
    assert result.summary_table[0] == {
        '% time': row.split()[0], 'seconds': row.split()[1],
        'usecs/call': row.split()[2], 'calls': row.split()[3],
        'errors': errors, 'syscall': row.split()[-1],
    }
