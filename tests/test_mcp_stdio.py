"""Synthetic subprocess coverage for JSON-RPC stdout and library log routing."""
import json
import os
import subprocess
import sys


def test_store_import_and_startup_keep_stdout_clean(tmp_path):
    code = "from knowledge_graph.services.graph_store import KnowledgeGraphStore; import json,sys; KnowledgeGraphStore(sys.argv[1], enable_snapshots=False); print(json.dumps({'ok':True}))"
    result = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=10, check=True)
    assert json.loads(result.stdout) == {"ok": True}
    assert "KnowledgeGraphStore initialized" in result.stderr


def test_mcp_stdio_initializes_and_calls_export_policy(tmp_path):
    code = '''
import asyncio,json,os,sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
async def main():
    params = StdioServerParameters(command=sys.executable,args=['-B','-m','knowledge_graph.mcp_server'],env=dict(os.environ))
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            result = await session.call_tool('kg_policy', {})
            print(json.dumps(result.model_dump(mode='json')))
asyncio.run(main())
'''
    environment = dict(os.environ, KG_DATA_DIR=str(tmp_path), KG_EXPORT_ONLY="1")
    result = subprocess.run([sys.executable, "-B", "-c", code], env=environment, capture_output=True, text=True, timeout=15, check=True)
    payload = json.loads(result.stdout)
    assert not payload["isError"]
    assert "approved_exports" in json.dumps(payload)
