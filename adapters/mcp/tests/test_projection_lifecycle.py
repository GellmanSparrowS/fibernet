"""Projection publication uses the shared staged artifact lifecycle."""
import copy
from pathlib import Path
from unittest.mock import patch

import anyio
import pytest

from fibernet_mcp import server
from fibernet_mcp.safety import ContractError, canonical, read_at, strict_json
from fibernet_mcp.projection import projection_request
from projection_fixtures import fixture_bytes
from test_projection import configure


def test_staged_projection_reply_cancellation_dedup_and_integrity(tmp_path):
    source=fixture_bytes('projection_native31.json')
    report=strict_json(fixture_bytes('projection_report.json'))
    _,output,_,_=configure(tmp_path,[(source,'m','lab_xyz')])
    store=server.ManagedStore(str(tmp_path/'manifest.json'),str(output))
    gid=next(iter(store.graphs))
    request=projection_request(store.graphs[gid],{'geometry_policy':'endpoint_chord'})
    identity=store.core_provenance
    async def exercise():
        before=(dict(store.artifacts),set(store.files),store.total)
        def reject(reply): raise ValueError('controlled reply rejection')
        with pytest.raises(ValueError,match='controlled reply rejection'):
            await server.publish_projection(store,gid,report,identity,request,reject)
        assert (store.artifacts,store.files,store.total)==before
        with anyio.CancelScope() as scope:
            with anyio.CancelScope(shield=True):
                scope.cancel()
                await anyio.sleep(0)
            await server.publish_projection(store,gid,report,identity,request,lambda reply:None)
            pytest.fail('pending cancellation must win before publication')
        assert scope.cancelled_caught
        assert (store.artifacts,store.files,store.total)==before
        # Fail a real bounded write after opening its owned output.
        files=set(Path(store.path).iterdir())
        with patch('fibernet_mcp.safety.os.fsync',side_effect=OSError('controlled sync failure')):
            with pytest.raises(OSError,match='controlled sync failure'):
                await server.publish_projection(store,gid,report,identity,request,lambda reply:None)
        assert set(Path(store.path).iterdir())==files
        assert (store.artifacts,store.files,store.total)==before
        checked=[]
        fresh_commits=[]
        first_commit=store.commit_artifact
        with anyio.CancelScope() as fresh_scope:
            def fresh_commit_then_cancel(stage):
                assert not stage.existing and not store.artifacts
                descriptor=first_commit(stage)
                fresh_commits.append(copy.deepcopy(descriptor))
                fresh_scope.cancel()
                return descriptor
            with patch.object(store,'commit_artifact',side_effect=fresh_commit_then_cancel):
                reply=await server.publish_projection(store,gid,report,identity,request,lambda r:checked.append(copy.deepcopy(r)))
                await anyio.lowlevel.checkpoint()
                pytest.fail('fresh result delivery must observe cancellation')
        assert fresh_scope.cancelled_caught
        assert checked==[reply] and reply['status']=='completed'
        assert fresh_commits==[reply['artifact']] and len(store.artifacts)==1
        aid=reply['artifact']['artifact_id'];saved=canonical(report)
        assert read_at(store.fd,aid)==saved
        original=copy.deepcopy(reply['artifact']);total=store.total
        reply['artifact']['producer']['adapter_version']='caller mutation'
        assert store.info(aid)==original
        with pytest.raises(ValueError,match='controlled reply rejection'):
            await server.publish_projection(store,gid,report,identity,request,reject)
        assert store.total==total and len(store.artifacts)==1
        repeated=await server.publish_projection(store,gid,report,identity,request,lambda r:None)
        assert repeated['artifact']==original and store.total==total
        # Deliver cancellation at the boundary after the real synchronous commit,
        # before the caller can finish delivering its successful result.
        commits=[]
        original_commit=store.commit_artifact
        with anyio.CancelScope() as scope:
            def commit_then_cancel(stage):
                descriptor=original_commit(stage)
                commits.append(copy.deepcopy(descriptor))
                scope.cancel()
                return descriptor
            with patch.object(store,'commit_artifact',side_effect=commit_then_cancel):
                completed=await server.publish_projection(store,gid,report,identity,request,lambda r:None)
                assert completed['artifact']==original
                await anyio.lowlevel.checkpoint()
                pytest.fail('cancellation must interrupt result delivery')
        assert scope.cancelled_caught and commits==[original]
        assert store.info(aid)==original and read_at(store.fd,aid)==saved
        # Staging rejects mutated pre-existing bytes, leaving the registry intact.
        target=Path(store.path)/aid;target.write_bytes(saved+b' ')
        with pytest.raises(ContractError,match='artifact integrity failure'):
            await server.publish_projection(store,gid,report,identity,request,lambda r:None)
        assert len(store.artifacts)==1 and store.total==total
        target.write_bytes(saved)
        assert store.info(aid)==original
    try:anyio.run(exercise)
    finally:store.close()


@pytest.mark.parametrize('collision',['output','request'])
def test_projection_supervisor_preserves_nonowned_collision(tmp_path,collision):
    _,output,_,_=configure(tmp_path,[(fixture_bytes('projection_native31.json'),'m','lab_xyz')])
    store=server.ManagedStore(str(tmp_path/'manifest.json'),str(output))
    gid=next(iter(store.graphs));request=projection_request(store.graphs[gid],{'geometry_policy':'endpoint_chord'})
    supervisor=server.Supervisor(store)
    class Fixed:
        hex='f'*32
    filename=('worker-' if collision=='output' else 'request-')+Fixed.hex
    target=Path(store.path)/filename;target.write_bytes(b'preexisting owner')
    try:
        with patch('fibernet_mcp.server.uuid.uuid4',return_value=Fixed()),patch('fibernet_mcp.server.subprocess.Popen',side_effect=AssertionError('must not launch')):
            with pytest.raises((ContractError,FileExistsError)):
                anyio.run(supervisor.run,'fiber_projection_report',gid,request)
        assert target.read_bytes()==b'preexisting owner'
        assert not store.artifacts and supervisor.active is None
    finally:
        # This test, not ManagedStore, owns the deliberately colliding file.
        target.unlink();store.close()
