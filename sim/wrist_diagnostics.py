"""Read-only task inspection; write wrist reports/scenes into a NEW directory.

python -m TemplatePattern_final.sim.wrist_diagnostics --stage2-dir TASK --output-dir NEW
Body OBJ is metres; GarmentCode simulation/boxmesh OBJ is centimetres.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .wrist_placement import read_obj


def cuff_indices(spec, labels, side):
    names = {n for n in spec['pattern']['panels'] if 'cuff' in n and
             n.endswith('_right') == (side == 'right')}
    seams = {f'stitch_{i}' for i, s in enumerate(spec['pattern']['stitches'])
             if any(e['panel'] in names for e in s[:2])}
    return np.asarray([i for i, label in enumerate(labels) if label in names or
                       any(s in seams for s in label.split(','))], dtype=int)


def triangle_intersections(cloth, cloth_faces, body, body_faces):
    """Exact triangle tests after AABB broad phase, including coplanar hits."""
    import igl
    triangles = body[body_faces]
    lo, hi = triangles.min(axis=1), triangles.max(axis=1)
    hits, pairs = [], 0
    for i, face in enumerate(cloth_faces):
        tri = cloth[face]
        possible = np.flatnonzero(np.all(hi >= tri.min(0)-1e-10, axis=1) &
                                  np.all(lo <= tri.max(0)+1e-10, axis=1))
        hit_face = False
        for j in possible:
            hit, *_ = igl.tri_tri_intersection_test_3d(*(v.reshape(1, 3) for v in [*tri, *triangles[j]]))
            if hit:
                hit_face = True
                pairs += 1
        if hit_face:
            hits.append(i)
    return {'cloth_faces_intersecting_body': len(hits), 'triangle_pair_count': pairs,
            'local_face_indices': hits}


def inspect(stage2_dir, output_dir):
    import igl
    import trimesh
    root, out = Path(stage2_dir).resolve(), Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite diagnostic results: {out}')
    spec = json.loads((root/'input/garment_specification.json').read_text())
    body, body_faces = read_obj(root/'body/body.obj')
    mesh = trimesh.Trimesh(body, body_faces, process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError('Signed penetration metrics require a closed, outward-oriented body')
    from .tpose_placement import BodyPosePlacement
    frames = BodyPosePlacement()._compute_landmarks(root/'body', 'body')['wrists']
    labels = (root/'simulation/sim_segmentation.txt').read_text().splitlines()
    out.mkdir(parents=True, exist_ok=True)
    from .sewing_scene import export_scene
    export_scene(root, out/'initial_sewing')
    report = {'units': 'millimetres for distances', 'anatomical_sides': True,
              'penetration_tolerance_mm': 0.1, 'states': {}}
    for state, filename in [('initial', 'boxmesh.obj'), ('final', 'sim.obj')]:
        source = root/'simulation'/filename
        if not source.exists():
            continue
        cloth, faces = read_obj(source)
        cloth = cloth/100.
        if len(cloth) != len(labels) or not np.isfinite(cloth).all():
            raise ValueError('Invalid geometry or per-vertex segmentation mismatch')
        sides = {}
        for side in ('left', 'right'):
            ids = cuff_indices(spec, labels, side)
            if not len(ids):
                continue
            distances = igl.signed_distance(cloth[ids], body, body_faces)[0]*1000.
            mask = np.isin(faces, ids).any(axis=1)
            cuff_faces = faces[mask]
            intersection = triangle_intersections(cloth, cuff_faces, body, body_faces)
            sides[side] = {'vertex_count_including_seams': len(ids),
                           'inside_vertex_count': int((distances < -0.1).sum()),
                           'max_penetration_mm': float(max(0., -distances.min())),
                           'median_signed_clearance_mm': float(np.median(distances)),
                           'inside_vertex_ids': ids[distances < -0.1].tolist(),
                           'wrist_center_cm': frames[side]['center_cm'],
                           'triangle_intersections': intersection}
            _closeup(out/f'{state}_{side}.png', body, body_faces, cloth, cuff_faces,
                     cloth[ids[distances < -0.1]], frames[side], f'{state}: anatomical {side}')
        report['states'][state] = sides
        _scene(out/f'{state}_body_cloth_seams.obj', body, body_faces, cloth, faces, labels)
    (out/'wrist_report.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'README.txt').write_text('人体与衣服场景统一为米。左右均为人体解剖侧。\n'
        'initial_sewing 为未合并裁片的实际端点缝合线；initial/final_body_cloth_seams 为网格场景，红线标出缝合顶点间的网格边。\n'
        '近景红点为人体内部超过 0.1 mm 的袖口/缝合边顶点；两视图用于观察深度，不代替三维检查。\n'
        '三角相交统计包括接触袖口顶点的边界三角面，不等同引擎全衣相交计数。\n', encoding='utf-8')
    return report


def _scene(path, body, body_faces, cloth, faces, labels):
    lines = ['mtllib diagnostic.mtl', 'o body', 'usemtl body']
    lines += ['v '+' '.join(map(str, v)) for v in body]
    lines += ['f '+' '.join(str(i+1) for i in f) for f in body_faces]
    lines += ['o cloth', 'usemtl cloth']
    lines += ['v '+' '.join(map(str, v)) for v in cloth]
    lines += ['f '+' '.join(str(i+len(body)+1) for i in f) for f in faces]
    lines += ['o seam_edges', 'usemtl seams']
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    for a, b in edges:
        if set(labels[a].split(',')) & set(labels[b].split(',')) and labels[a].startswith('stitch_') and labels[b].startswith('stitch_'):
            lines.append(f'l {a+len(body)+1} {b+len(body)+1}')
    path.write_text('\n'.join(lines)+'\n')
    (path.parent/'diagnostic.mtl').write_text('newmtl body\nKd .3 .3 .3\nnewmtl cloth\nKd .6 .75 .9\nd .65\nnewmtl seams\nKd 1 .1 .1\n')


def _closeup(path, body, body_faces, cloth, cuff_faces, inside, frame, title):
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-templatepattern-final')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    center = np.array(frame['center_cm'])/100
    axes = np.array([frame['axis'], frame['transverse'], frame['front_normal']]).T
    b = (body-center)@axes*1000
    c = (cloth-center)@axes*1000
    points = (inside-center)@axes*1000
    local = body_faces[np.linalg.norm(body[body_faces].mean(1)-center, axis=1)<.22]
    fig, panels = plt.subplots(1, 2, figsize=(10, 5))
    for ax, dims in zip(panels, ([0, 1], [0, 2])):
        ax.add_collection(PolyCollection(b[local][:, :, dims], facecolors='#888888', edgecolors='none', alpha=.22))
        ax.add_collection(PolyCollection(c[cuff_faces][:, :, dims], facecolors='#88bbee', edgecolors='#446688', linewidths=.2, alpha=.3))
        if len(points):
            ax.scatter(points[:, dims[0]], points[:, dims[1]], c='red', s=10)
        ax.set_xlim(-180, 80); ax.set_ylim(-100, 100); ax.set_aspect('equal')
        ax.set_xlabel('forearm axis (mm; wrist = 0)'); ax.set_ylabel('transverse (mm)' if dims[1] == 1 else 'depth (mm)')
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage2-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.stage2_dir, args.output_dir)
    print(json.dumps({state: {side: {k: v for k, v in values.items() if k in ('inside_vertex_count', 'max_penetration_mm')}
                             for side, values in sides.items()} for state, sides in result['states'].items()}, indent=2))
