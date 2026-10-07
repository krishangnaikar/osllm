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
