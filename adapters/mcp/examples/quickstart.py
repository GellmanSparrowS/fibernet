"""Validate the matching local FiberNet MCP installation using a known graph."""
import argparse
import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "voronoi_4340.json"
FIXTURE_SHA = "fb40544438cd4e7ca9bc60d8f10c6363c169b0aa7ce99f4e0a3d45b64629c584"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


async def run(fixture):
    import fibernet
    import fibernet_mcp
    from fibernet.analysis.topology_metrics import analyze_topology
    from fibernet.core.structure_graph import StructureGraph

    expected = json.loads(Path(fibernet_mcp.__file__).with_name("core_manifest.json").read_bytes())
    digest = hashlib.sha256(json.dumps(expected,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
    _require(hashlib.sha256(fixture.read_bytes()).hexdigest() == FIXTURE_SHA,"Fixture does not match this checkout")
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        output = work / "output"
        output.mkdir()
        operator = work / "manifest.json"
        operator.write_text(json.dumps({
            "source_tree": "6eabb154cd37d95c7c447a568b1b737c3ec883c8",
            "graphs": [{"path":str(fixture), "sha256":FIXTURE_SHA,
                        "length_unit":"unspecified","frame_id":"unknown"}],
        }),encoding="utf-8")
        server = StdioServerParameters(
            command=sys.executable,
            args=["-I","-m","fibernet_mcp.server","--manifest",str(operator),"--output-root",str(output)],
            env={"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","PYTHONDONTWRITEBYTECODE":"1",
                 "MPLCONFIGDIR":str(work / "mpl"), "XDG_CACHE_HOME":str(work / "cache")},
            cwd=directory,
        )
        with anyio.fail_after(25):
            async with stdio_client(server) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    init = await session.initialize()
                    _require(init.protocolVersion == "2025-11-25","Unexpected MCP protocol")
                    resource = await session.read_resource("fibernet://provenance/1")
                    actual = json.loads(resource.contents[0].text)
                    _require(actual["canonical_core_manifest_sha256"] == digest,"Core provenance differs")
                    _require(actual["source_file_count"] == len(expected),"Core source inventory differs")
                    catalog = await session.call_tool("fibernet_catalog",{})
                    _require(not catalog.isError,"Catalog request failed")
                    graph,=catalog.structuredContent["graphs"]
                    _require((graph["node_count"],graph["edge_count"])==(4340,5073),"Unexpected fixture graph")
                    topology=await session.call_tool("fibernet_analyze_topology",{"graph_id":graph["graph_id"]})
                    _require(not topology.isError,"Topology request failed")
                    artifact=topology.structuredContent["artifact"]
                    raw=bytearray()
                    offset=0
                    for _ in range(256):
                        response=await session.call_tool("fibernet_artifact_read",{
                            "artifact_id":artifact["artifact_id"],"offset":offset,"length":32768,
                        })
                        _require(not response.isError,"Artifact read failed")
                        chunk=response.structuredContent
                        _require(chunk["offset"]==offset and chunk["total_bytes"]==artifact["byte_size"]
                                 and chunk["sha256"]==artifact["sha256"], "Artifact chunk identity differs")
                        raw.extend(base64.b64decode(chunk["data"],validate=True))
                        _require(chunk["next_offset"]==len(raw),"Artifact offset does not match")
                        if chunk["eof"]:
                            break
                        _require(chunk["next_offset"] > offset,"Artifact stream did not advance")
                        offset=chunk["next_offset"]
                    else:
                        raise RuntimeError("Too many artifact chunks")
                    _require(len(raw)==artifact["byte_size"],"Artifact byte length differs")
                    _require(hashlib.sha256(raw).hexdigest()==artifact["sha256"],"Artifact hash does not match")
                    direct=asdict(analyze_topology(StructureGraph.load_json(str(fixture))))
                    _require(json.loads(raw)==json.loads(json.dumps(direct)),"MCP topology differs from the native SDK")
    print(json.dumps({"fibernet":fibernet.__version__,"protocol":init.protocolVersion,
        "source_files":len(expected),"provenance_sha256":digest,"nodes":graph["node_count"],
        "edges":graph["edge_count"],"artifact_bytes":len(raw),"artifact_sha256":artifact["sha256"],
        "native_topology_equal":True}))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    anyio.run(run,FIXTURE.resolve(strict=True))
