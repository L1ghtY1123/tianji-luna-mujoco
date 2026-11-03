"""Render static catalog references directly from verified metric CAD triangles."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from luna_mujoco.cad_perception import CadCatalog


def rasterize(triangles, normals, base, radius, elevation, azimuth, pixels=512):
    """Orthographic CPU triangle rasterization with an explicit depth buffer."""
    elevation, azimuth = np.deg2rad([elevation, azimuth])
    towards_camera = np.array([np.cos(elevation) * np.cos(azimuth),
                               np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    right = np.cross([0., 0., 1.], towards_camera)
    right /= np.linalg.norm(right)
    up = np.cross(towards_camera, right)
    basis = np.column_stack((right, up, towards_camera))
    camera = triangles @ basis
    projected = camera[..., :2] * (pixels * .43 / radius) + pixels / 2
    projected[..., 1] = pixels - projected[..., 1]
    light = towards_camera + .4 * right + .6 * up
    light /= np.linalg.norm(light)
    # Two-sided material shading preserves the source's open-surface geometry.
    colors = np.clip((.35 + .65 * np.abs(normals @ light))[:, None] * base * 255, 0, 255).astype(np.uint8)
    image = np.full((pixels, pixels, 3), 255, dtype=np.uint8)
    depth = np.full((pixels, pixels), -np.inf)
    for xy, triangle, color in zip(projected, camera, colors, strict=True):
        low = np.maximum(np.floor(xy.min(axis=0)).astype(int), 0)
        high = np.minimum(np.ceil(xy.max(axis=0)).astype(int), pixels - 1)
        if np.any(high < low):
            continue
        a, b, c = xy
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-12:
            continue
        rows, columns = np.mgrid[low[1]:high[1] + 1, low[0]:high[0] + 1]
        x, y = columns + .5, rows + .5
        first = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
        second = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
        third = 1 - first - second
        z = first * triangle[0, 2] + second * triangle[1, 2] + third * triangle[2, 2]
        selected = (first >= -1e-8) & (second >= -1e-8) & (third >= -1e-8) & (z > depth[rows, columns])
        image[rows[selected], columns[selected]] = color
        depth[rows[selected], columns[selected]] = z[selected]
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh-key', help='Render only the selected catalog asset')
    args = parser.parse_args()
    catalog = CadCatalog.from_file()
    path = ROOT / 'resources/cad_catalog.json'
    document = json.loads(path.read_text(encoding='utf-8'))
    for record in document['assets']:
        if args.mesh_key and record['mesh_key'] != args.mesh_key:
            continue
        geometry = catalog.geometry(record['mesh_key'])
        vertices = geometry['vertices_m'] * 1000
        triangles = vertices[geometry['faces']]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        blue = record['mesh_key'] == 'blue_box'
        base = np.array([.15, .42, .78] if blue else [.36, .39, .43])
        size = np.ptp(vertices, axis=0)
        radius = float(size.max()) * .6
        figure = plt.figure(figsize=(14 if blue else 18, 5), facecolor='white')
        figure.suptitle('BLUE MATERIAL BIN' if blue else 'BLACK THREADED WORKPIECE',
                        fontsize=20, fontweight='bold', y=.97)
        angles = ([(30, -55), (30, 125), (78, -45)] if blue else
                  [(25, -55), (25, 125), (5, 25), (78, -45)])
        labels = (['Three-quarter', 'Opposite three-quarter', 'Top'] if blue else
                  ['Three-quarter', 'Opposite three-quarter', 'Side', 'Near-overhead'])
        for index, (elevation, azimuth) in enumerate(angles, start=1):
            axes = figure.add_subplot(1, len(angles), index)
            axes.imshow(rasterize(triangles, normals, base, radius, elevation, azimuth))
            axes.set_axis_off()
            axes.set_title(labels[index - 1], fontsize=12, y=.02)
        dimensions = ' x '.join(f'{value:g}' for value in size)
        figure.text(.5, .07, f'CAD dimensions: {dimensions} mm', ha='center', fontsize=13)
        figure.text(.5, .02, 'Static CAD reference | original axes retained | material colour configured | no scene pose',
                    ha='center', fontsize=10, color='#444444')
        figure.subplots_adjust(left=.01, right=.99, bottom=.13, top=.89, wspace=-.05)
        output = ROOT / 'resources' / record['preview_path']
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=130, facecolor='white', metadata={'Software': 'Matplotlib CAD triangles'})
        plt.close(figure)
        record['preview_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
        print(json.dumps({'asset_id': record['asset_id'], 'path': str(output),
                          'sha256': record['preview_sha256'], 'triangles': len(triangles)}))
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
