"""Rigid panel placement relative to the reference and target body.

Never changes vertices, edges, stitch endpoints, directions or material.
Reference-body input is an identity transform, preserving the original layout.
"""
from __future__ import annotations
from math import atan2, cos, sin, hypot, degrees
from TemplatePattern_final.shared.io import ROOT, read_json


def place_relative_to_body(spec, target, style, pose, relation):
    references = read_json(ROOT/'assets/placement_reference.json')
    key = style + (':tpose' if style == 'short_sleeve' or pose == 'tpose' else ':a45')
    reference = references[key]
    torso, old_torso = target['torso'], reference['torso']
    height_ratio = target['height_cm']/reference['height_cm']
    width = lambda body: abs(body['segments']['left']['shoulder'][0]-body['segments']['right']['shoulder'][0])
    width_ratio = width(target)/width(reference)
    for name, panel in spec['pattern']['panels'].items():
        original_t = list(panel['translation']); original_r = list(panel['rotation'])
        t, r = list(original_t), list(original_r)
        if 'sleeve' in name or 'cuff' in name:
            side = 'right' if name.endswith('_right') else 'left'
            old, new = reference['segments'][side], target['segments'][side]
            old_axis = [old['hand_root'][i]-old['shoulder'][i] for i in (0,1)]
            new_axis = [new['hand_root'][i]-new['shoulder'][i] for i in (0,1)]
            old_length, new_length = hypot(*old_axis), hypot(*new_axis)
            old_angle, new_angle = atan2(old_axis[1],old_axis[0]), atan2(new_axis[1],new_axis[0])
            dx, dy = t[0]-old['shoulder'][0], t[1]-old['shoulder'][1]
            axial = (dx*cos(old_angle)+dy*sin(old_angle))*new_length/old_length
            perpendicular = -dx*sin(old_angle)+dy*cos(old_angle)
            t[0] = new['shoulder'][0]+axial*cos(new_angle)-perpendicular*sin(new_angle)
            t[1] = new['shoulder'][1]+axial*sin(new_angle)+perpendicular*cos(new_angle)
            r[2] += degrees(new_angle-old_angle)
            old_bounds, new_bounds = old['bounds_cm'], new['bounds_cm']
            region = side+'_arm'
        else:
            collar = 'collar' in name
            anchor = 'neck_center' if collar else 'chest_center'
            old_center, new_center = old_torso[anchor], torso[anchor]
            t[0] = new_center[0]+(t[0]-old_center[0])*width_ratio
            t[1] = new_center[1]+(t[1]-old_center[1])*(1. if collar else height_ratio)
            bound_key = 'neck_bounds_cm' if collar else 'bounds_cm'
            old_bounds, new_bounds = old_torso[bound_key], torso[bound_key]
            region = 'neck' if collar else 'torso'
        # Preserve calibrated front/back clearance instead of using fixed Z.
        surface = 5 if original_t[2] >= (old_bounds[2]+old_bounds[5])/2 else 2
        t[2] = new_bounds[surface]+original_t[2]-old_bounds[surface]
        panel['translation'] = [round(v,6) for v in t]
        panel['rotation'] = [round(v,6) for v in r]
        relation['piece_placement'][name] = {
            'region':region, 'before':{'translation':original_t,'rotation':original_r},
            'after':{'translation':panel['translation'],'rotation':panel['rotation']},
        }
    relation['strategy'] = 'reference_body_relative_rigid_panels'
    relation['reference_key'] = key
    relation['height_ratio'] = height_ratio
    relation['shoulder_width_ratio'] = width_ratio
