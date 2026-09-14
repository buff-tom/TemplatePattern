"""Transfer Stage1 size changes to semantically corresponding reference panels.

The reference owns panel splitting, boundary discretisation and sewing indices.
Stage1 owns parent-piece dimensions. This size-only adapter deliberately does
not claim to transfer arbitrary changes to boundary curvature or topology.
"""
from __future__ import annotations
import numpy as np
from TemplatePattern_final.shared.io import ROOT, read_json


def parent_piece(panel):
    if panel in ('front_body_left', 'front_body_right', 'back_body', 'back_yoke'):
        return panel
    for prefix in ('long_sleeve', 'short_sleeve', 'cuff', 'collar_stand', 'collar_fall'):
        if panel.startswith(prefix+'_'):
            return prefix
    raise ValueError(f'No Stage1 semantic parent for reference panel {panel!r}')


def dimensions(points, label):
    vertices = np.asarray(points, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3 or not np.isfinite(vertices).all():
        raise ValueError(f'{label}: expected a finite 2D polygon')
    span = np.ptp(vertices, axis=0)
    if np.any(span <= 1e-6):
        raise ValueError(f'{label}: degenerate width/height')
    return span


def transfer_sizes(spec, pattern):
    style = pattern['style']
    template_path = ROOT/'templates'/style/f'{style}_shirt_template_M.json'
    template = read_json(template_path)
    anchors = {p['id']: p for p in template['pieces']}
    pieces = pattern.get('pieces')
    if not isinstance(pieces, dict) or not pieces:
        raise ValueError('Stage2 conversion requires Stage1 pieces; fixed reference fallback is disabled')
    changes = {}
    # Validate all required parents before modifying the copied reference.
    scales = {}
    for name in spec['pattern']['panels']:
        parent = parent_piece(name)
        if parent not in pieces or parent not in anchors:
            raise ValueError(f'{name}: missing semantic parent {parent!r}')
        if parent not in scales:
            target = dimensions(pieces[parent].get('boundary'), f'Stage1 {parent}')
            base = dimensions(anchors[parent].get('boundary'), f'M anchor {parent}')
            scales[parent] = (target/base, target, base)
    for name, panel in spec['pattern']['panels'].items():
        parent = parent_piece(name)
        scale, target, base = scales[parent]
        vertices = np.asarray(panel['vertices'], dtype=float)
        before = dimensions(vertices, name)
        # Scale in reference local axes about the reference origin. Keep every
        # edge endpoint/index and every seam entry byte-for-byte equivalent.
        panel['vertices'] = (vertices*scale).tolist()
        changes[name] = {'stage1_parent': parent, 'scale_xy': scale.tolist(),
                         'stage1_parent_dimensions_mm': target.tolist(),
                         'anchor_parent_dimensions_mm': base.tolist(),
                         'reference_dimensions_cm': before.tolist(),
                         'output_dimensions_cm': dimensions(panel['vertices'], name).tolist()}
    return {'method': 'semantic_parent_size_transfer',
            'anchor': str(template_path.relative_to(ROOT)),
            'scope': 'parent width/height ratios; reference splitting and local curves retained',
            'panels': changes}
