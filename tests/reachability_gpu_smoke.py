"""Pinned real GPU, both arms, recorded failure points and an impossible point."""
import json

from tianji_curobo.curobo_service import materialize_urdf

from luna_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation, sdk_identity


def main():
    urdf = materialize_urdf(ROOT / 'assets/marvin_urdf/marvin_with_grippers.urdf',
        marvin_meshes=ROOT / 'assets/marvin_urdf/meshes', rag_meshes=ROOT / 'assets/marvin_urdf/rag100_050/meshes')
    runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
    recorded = [[.5025, .3514, .6568], [.4948, .0237, .6002]]
    report = {'sdk': sdk_identity()[0], 'arms': {}}
    for arm in ('A', 'B'):
        payload = {'arm': arm, 'positions_world_m': recorded}
        first = run_operation(runtime, '/tools/evaluate_reachability', payload)
        cached = runtime._reachability_solvers[arm]
        warm = run_operation(runtime, '/tools/evaluate_reachability', payload)
        assert runtime._reachability_solvers[arm] is cached
        assert warm['items'][1]['reachable']
        if arm == 'A':
            assert warm['items'][0]['reachable']
        assert all(item['reference_offset_m'] == [0., 0., .185] for item in warm['items'])
        impossible = run_operation(runtime, '/tools/evaluate_reachability',
            {'arm': arm, 'positions_world_m': [[5., 0., 2.]]})
        assert len(impossible['items']) == 1 and not impossible['items'][0]['reachable']
        assert impossible['items'][0]['position_error_m'] > 1.
        report['arms'][arm] = {'cold': first, 'warm': warm, 'outside_workspace': impossible}
    assert runtime.geometry_scene is None and not runtime._backends
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
