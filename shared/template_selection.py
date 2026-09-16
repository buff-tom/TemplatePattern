"""Capacity-first selection, with virtual anchors and wooden-barrel fallback.

The bottleneck and range-normalized distance follow the v4 exec_7 selector.
Within range we also require all six upper capacities, as the input contract
requests, instead of allowing a smaller average-nearest anchor.
"""
from __future__ import annotations
from math import sqrt

WEIGHTS = dict(height=.8, chest=1.5, waist=1., hip=1., shoulder_width=1.2, arm_length=1.)


def select_body_reference_template_with_fallback(body, candidates, weights=None):
    weights = weights or WEIGHTS
    supported_range = {
        k: [min(float(r[k]) for r in candidates.values()), max(float(r[k]) for r in candidates.values())]
        for k in weights
    }
    spans = {k: max(supported_range[k][1]-supported_range[k][0], 1.) for k in weights}
    distance = lambda r: sqrt(sum(w*((body[k]-r[k])/spans[k])**2 for k,w in weights.items()))
    normal = min(candidates, key=lambda n: distance(candidates[n]))
    outside = []
    for k in weights:
        lo, hi = supported_range[k]
        if not lo <= body[k] <= hi:
            outside.append({'key': k, 'direction': 'low' if body[k] < lo else 'high', 'value': body[k], 'range': [lo,hi], 'normalized_outside': max(lo-body[k],body[k]-hi)/spans[k]})
    # First select an anchor large enough in every dimension. This is the
    # equivalent of taking the maximum required size on monotone size tables.
    covering = {n:r for n,r in candidates.items() if all(r[k] >= body[k]-1e-9 for k in weights)}
    constraints = outside or [{'key':k, 'direction':'high', 'value':body[k]} for k in weights]
    gap = lambda r,c: (r[c['key']]-c['value'])/spans[c['key']] * (1 if c['direction']=='high' else -1)
    score = lambda r: min(gap(r,c) for c in constraints)
    if covering and not outside:
        selected = min(covering, key=lambda n: distance(covering[n]))
        strategy, status = 'capacity_covering', 'all_constraints_covered'
    else:
        # Preserve the historical out-of-range bottleneck rule. Distance is
        # the tie-breaker, never an exception or a no-template outcome.
        best = max(score(r) for r in candidates.values())
        eligible = {n:r for n,r in candidates.items() if abs(score(r)-best) <= 1e-9}
        selected = min(eligible, key=lambda n: distance(eligible[n]))
        strategy, status = 'covering_fallback', 'best_bottleneck_available'
    ref = candidates[selected]
    rows = [{**c, 'selected_anchor_value':ref[c['key']], 'normalized_capacity_gap':gap(ref,c), 'covered_by_selected_anchor':gap(ref,c)>=-1e-9} for c in constraints]
    driver = min(rows, key=lambda r:r['normalized_capacity_gap'])
    return {'nearest_size':ref.get('shape_prior_size',selected), 'nearest_anchor':selected,
            'distance':distance(ref), 'selection_strategy':strategy,
            'normal_nearest_anchor':normal, 'normal_nearest_size':candidates[normal].get('shape_prior_size',normal),
            'fallback_driver':driver if strategy=='covering_fallback' else None,
            'fallback_constraints':{'status':status,'bottleneck_key':driver['key'],'constraints':rows},
            'selected_capacity_mm':{k:ref[k] for k in weights},
            'supported_range_mm':supported_range,
            'input_outside_supported_range':bool(outside),
            'extrapolation_level':'inside_range' if not outside else 'near_extrapolation' if max(c['normalized_outside'] for c in outside)<=.15 else 'far_extrapolation'}


def select_template(body, references):
    return select_body_reference_template_with_fallback(body, {**references['sizes'], **references.get('virtual_anchors',{})})
