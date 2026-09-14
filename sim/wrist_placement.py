"""Anatomical wrist frames and rigid sleeve/cuff placement (centimetres).

No rest geometry or sewing data is edited. All anchors come from the current
body and the existing sleeve/cuff stitch, not panel origins or image sides.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def unit(v):
    v = np.asarray(v, dtype=float)
    length = np.linalg.norm(v)
    if not np.isfinite(length) or length < 1e-8:
        raise ValueError('Degenerate wrist coordinate frame')
    return v / length


def read_obj(path):
    vertices, faces = [], []
    for line in path.open():
        fields = line.split()
        if fields and fields[0] == 'v':
            vertices.append([float(x) for x in fields[1:4]])
        elif fields and fields[0] == 'f':
            ids = [int(x.split('/')[0]) - 1 for x in fields[1:]]
            faces.extend([[ids[0], ids[i], ids[i+1]] for i in range(1, len(ids)-1)])
    return np.asarray(vertices), np.asarray(faces, dtype=int)


def wrist_frames(vertices_cm, faces, segmentation, segments):
    vertices_cm = np.asarray(vertices_cm)
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                             faces[:, [2, 0]]]), axis=1), axis=0)
    result = {}
    for side in ('left', 'right'):
        hand = set(segmentation[side+'Hand'])
        arm = set(segmentation[side+'ForeArm'])
        boundary = [(vertices_cm[a]+vertices_cm[b])/2 for a, b in edges
                    if (a in hand and b in arm) or (b in hand and a in arm)]
        if len(boundary) < 3:
            raise ValueError(f'{side}: no valid hand/forearm boundary in body segmentation')
        ring = np.asarray(boundary)
        center = ring.mean(axis=0)
        axis = unit(center - segments[side]['forearm'])
        normal = unit(np.array([0., 0., 1.]) - axis*axis[2])
        transverse = unit(np.cross(-axis, normal))
        radial = np.c_[(ring-center)@transverse, (ring-center)@normal]
        hull = radial[ConvexHull(radial).vertices]
        perimeter = np.linalg.norm(hull - np.roll(hull, 1, axis=0), axis=1).sum()
        surface = vertices_cm[sorted(arm | hand | set(segmentation[side+'Arm']))]
        result[side] = dict(center_cm=center.tolist(), axis=axis.tolist(),
                            front_normal=normal.tolist(), transverse=transverse.tolist(),
                            boundary_points_cm=ring.tolist(), perimeter_cm=float(perimeter),
                            surface_points_cm=surface.tolist())
    return result


def local_vertices(panel):
    v = np.asarray(panel['vertices'], dtype=float)
    return np.c_[v, np.zeros(len(v))]


def world_vertices(panel):
    return Rotation.from_euler('xyz', panel['rotation'], degrees=True).apply(local_vertices(panel)) + panel['translation']


def place_wrists(spec, frames, relation, clearance_cm, body_vertices_cm=None):
    if not np.isfinite(clearance_cm) or clearance_cm <= 0:
        raise ValueError('Wrist clearance must be finite and positive')
    panels = spec['pattern']['panels']
    report = {'strategy': 'wrist_boundary_rigid', 'clearance_cm': clearance_cm, 'sides': {}}
    relation['wrist_placement'] = report
    for side, frame in frames.items():
        cuff_names = {n for n in panels if 'cuff' in n and n.endswith('_right') == (side == 'right')}
        if not cuff_names:
            continue
        axis, normal = np.array(frame['axis']), np.array(frame['front_normal'])
        center = np.array(frame['center_cm'])
        surface = np.array(frame['surface_points_cm'])
        connections = []
        for seam in spec['pattern']['stitches']:
            a, b = seam[:2]
            if a['panel'] in cuff_names and 'sleeve' in b['panel']:
                connections.append((a, b))
            elif b['panel'] in cuff_names and 'sleeve' in a['panel']:
                connections.append((b, a))
        if {c['panel'] for c, _ in connections} != cuff_names:
            raise ValueError(f'{side}: every cuff needs an existing sleeve attachment seam')
        circumference = 0.
        details = []
        for cuff_ref, sleeve_ref in connections:
            cuff, sleeve = panels[cuff_ref['panel']], panels[sleeve_ref['panel']]
            cv = local_vertices(cuff)
            ce = cv[cuff['edges'][cuff_ref['edge']]['endpoints']]
            anchor = ce.mean(axis=0)
            circumference += np.linalg.norm(ce[1]-ce[0])
            # The direction from cuff interior toward its sleeve seam is proximal.
            local_y = unit(anchor-cv.mean(axis=0))
            # Remove uneven boundary sampling bias along the straight seam.
            tangent = unit(ce[1]-ce[0])
            local_y = unit(local_y - tangent*np.dot(local_y, tangent))
            local_x = np.cross(local_y, [0., 0., 1.])
            length = float(np.max((anchor-cv)@local_y))
            front = 'front' in cuff_ref['panel']
            z = normal if front else -normal
            y = -axis
            x = unit(np.cross(y, z))
            rotation = np.column_stack((x, y, z)) @ np.column_stack((local_x, local_y, [0., 0., 1.])).T
            # Free cuff opening at the wrist; existing seam one cuff-length proximal.
            target = center - axis*length
            axial = (surface-center)@axis
            patch = surface[(axial >= -length-1.) & (axial <= 1.)]
            if not len(patch):
                raise ValueError(f'{side}: empty wrist surface patch')
            offset = float(np.max((patch-center)@z)) + clearance_cm
            target += z*offset
            before = {'translation': list(cuff['translation']), 'rotation': list(cuff['rotation'])}
            cuff['translation'] = (target-rotation@anchor).tolist()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                cuff['rotation'] = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True).tolist()
            # Swing the sleeve about the existing distal edge. Translating it
            # alone can put its proximal end through the upper arm/chest.
            sv = local_vertices(sleeve)
            se = sv[sleeve['edges'][sleeve_ref['edge']]['endpoints']]
            sleeve_anchor = se.mean(axis=0)
            sleeve_before = {'translation': list(sleeve['translation']), 'rotation': list(sleeve['rotation'])}
            body_points = surface if body_vertices_cm is None else np.asarray(body_vertices_cm)
            low, high = sv.min(0)-clearance_cm, sv.max(0)+clearance_cm
            # Also respect the proximal sleeve/body sewing interface. Choosing
            # only the smallest collision-free angle can crowd the shoulder.
            local_cap, target_cap = [], []
            for seam in spec['pattern']['stitches']:
                a, b = seam[:2]
                if b['panel'] == sleeve_ref['panel']:
                    a, b = b, a
                if a['panel'] != sleeve_ref['panel'] or 'sleeve' in b['panel'] or 'cuff' in b['panel']:
                    continue
                local_cap.extend(sv[sleeve['edges'][a['edge']]['endpoints']])
                other = panels[b['panel']]
                pair = world_vertices(other)[other['edges'][b['edge']]['endpoints']]
                target_cap.extend(pair if len(seam)>2 and seam[2]=='right_wrong' else pair[::-1])
            local_cap, target_cap = np.asarray(local_cap), np.asarray(target_cap)
            candidates = []
            base_target = target.copy()
            # A tilted sleeve may require slightly more clearance at its distal
            # edge than the cuff plane. Move BOTH connected pieces out together.
            for extra_clearance in np.arange(0., 2.01, .1):
                target = base_target + z*extra_clearance
                for swing in np.arange(0., 85.01, .5):
                    sr = Rotation.from_rotvec(x*np.deg2rad(swing)).as_matrix()@rotation
                    st = target-sr@sleeve_anchor
                    local_body = (body_points-st)@sr
                    mask = np.all(local_body[:, :2] >= low[:2], axis=1) & np.all(local_body[:, :2] <= high[:2], axis=1)
                    support = float(local_body[mask, 2].max()) if mask.any() else -float('inf')
                    if support <= -clearance_cm:
                        error = float(np.mean(np.sum((local_cap@sr.T+st-target_cap)**2, axis=1))) if len(local_cap) else float(swing)
                        candidates.append((error, float(swing), sr, st, support))
                if candidates:
                    break
            if not candidates:
                raise ValueError(f'{side}: sleeve cannot clear body with a rigid swing <=85 degrees; geometry unchanged')
            cap_error, swing, sr, st, support = min(candidates, key=lambda item: item[0])
            cuff['translation'] = (target-rotation@anchor).tolist()
            sleeve['translation'] = st.tolist()
            sleeve['rotation'] = Rotation.from_matrix(sr).as_euler('xyz', degrees=True).tolist()
            for name, panel, old in [(cuff_ref['panel'], cuff, before), (sleeve_ref['panel'], sleeve, sleeve_before)]:
                relation['piece_placement'][name]['wrist_adjustment'] = {'before': old, 'after': {
                    'translation': panel['translation'], 'rotation': panel['rotation']}}
                relation['piece_placement'][name]['after'] = {'translation': panel['translation'], 'rotation': panel['rotation']}
            new_se = world_vertices(sleeve)[sleeve['edges'][sleeve_ref['edge']]['endpoints']]
            new_ce = world_vertices(cuff)[cuff['edges'][cuff_ref['edge']]['endpoints']]
            details.append({'cuff': cuff_ref['panel'], 'sleeve': sleeve_ref['panel'],
                            'cuff_length_cm': length, 'surface_offset_cm': offset,
                            'extra_pair_clearance_cm': float(extra_clearance),
                            'sleeve_swing_degrees': float(swing),
                            'proximal_seam_rms_cm': float(np.sqrt(cap_error)) if len(local_cap) else None,
                            'sleeve_plane_clearance_cm': float(-support) if np.isfinite(support) else None,
                            'seam_center_gap_cm': float(np.linalg.norm(new_se.mean(0)-new_ce.mean(0))),
                            'seam_length_difference_cm': float(abs(np.linalg.norm(new_se[1]-new_se[0])-np.linalg.norm(new_ce[1]-new_ce[0])))})
        # Sum of the existing attachment widths is the closed cuff circumference.
        circumference = float(circumference)
        required = float(frame['perimeter_cm'] + 2*np.pi*clearance_cm)
        report['sides'][side] = {'wrist_center_cm': frame['center_cm'], 'axis': frame['axis'],
                                 'front_normal': frame['front_normal'], 'cuff_circumference_cm': circumference,
                                 'required_circumference_cm': required, 'geometry_sufficient': circumference >= required,
                                 'connections': details}
        if circumference < required:
            relation['wrist_placement'] = report
            raise ValueError(f'{side} cuff geometry insufficient: {circumference:.2f} cm < wrist plus clearance {required:.2f} cm; no geometry was resized')
    relation['wrist_placement'] = report
