"""One bounded official final-result fit per link; candidates require geometry review."""

from dataclasses import asdict

import numpy as np
import trimesh


def fit_motion_link(link_name, mesh, *, num_spheres, seed, iterations=200):
    """Fit the whole link once without combining snapshots or appending patches."""
    if type(num_spheres) is not int or num_spheres <= 0:
        raise ValueError("The per-link sphere budget must be a positive integer")
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("Iterations must be a positive integer")

    import torch
    from curobo.sphere_fit import fit_spheres_to_mesh

    prepared = mesh.copy()
    prepared.fill_holes()
    trimesh.repair.fix_normals(prepared)
    trimesh.repair.fix_inversion(prepared)
    trimesh.repair.fix_winding(prepared)
    np.random.seed(seed)
    torch.manual_seed(seed)
    result = fit_spheres_to_mesh(
        prepared, num_spheres=num_spheres, iterations=iterations, compute_metrics=True
    )
    centers = np.asarray(result.centers.tolist(), dtype=float)
    radii = np.asarray(result.radii.tolist(), dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or radii.shape != (len(centers),):
        raise ValueError(f"{link_name}: official final result has invalid sphere array shapes")
    if len(centers) > num_spheres:
        raise ValueError(f"{link_name}: official final result exceeds per-link budget {num_spheres}")
    if len(centers) == 0 or result.num_spheres != len(centers):
        raise ValueError(f"{link_name}: official final result has an invalid sphere count")
    if not np.isfinite(centers).all() or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError(f"{link_name}: official final result has invalid centers or radii")
    return {
        "link": link_name,
        "result_source": "official_final",
        "sphere_budget": num_spheres,
        "sphere_count": len(centers),
        "spheres": [
            {"center": center.tolist(), "radius": float(radius)} for center, radius in zip(centers, radii)
        ],
        "seed": seed,
        "api_parameters": {"num_spheres": num_spheres, "iterations": iterations, "compute_metrics": True},
        "fit_time_s": result.fit_time_s,
        "metrics": asdict(result.metrics) if result.metrics is not None else None,
        "debug_info": result.debug_info,
        "candidate_only": True,
        "geometry_reviewed": False,
    }
