"""Paired arclength resampling, avoiding microscopic seam subdivision slivers."""
from bisect import bisect_right
from math import ceil, dist


def build_dynamic_stitches(pattern, panels, lookup, builder):
    seams = [s for s in pattern['seams'] if s.get('target') == 'sew']
    allocated = builder._allocate_reused_edges(seams, lookup, builder._edge_lengths(pattern))
    replacements, owned, plan = {}, set(), []

    def chain(name, ids):
        vertices = panels[name]['vertices']
        if len(ids) < 2:
            raise ValueError(f'Missing semantic seam on {name}')
        forward = (ids[0]+1) % len(vertices) == ids[1]
        lengths = [0.]
        for a, b in zip(ids, ids[1:]):
            if (a + (1 if forward else -1)) % len(vertices) != b:
                raise ValueError(f'Non-contiguous semantic seam: {name}: {a}->{b}')
            key = (name, a if forward else b)
            if key in owned:
                raise ValueError(f'Boundary edge reused by multiple seams: {key}')
            owned.add(key)
            lengths.append(lengths[-1] + dist(vertices[a], vertices[b]))
        if lengths[-1] <= 1e-8:
            raise ValueError(f'Zero length seam: {name}')
        return vertices, lengths, forward

    for index, seam in enumerate(seams):
        sides = []
        for side in ('a', 'b'):
            name = seam[side]['piece']
            ids = allocated.get((index, side), [])
            if side == 'b' and seam.get('direction', 'same') == 'opposite':
                ids = list(reversed(ids))
            sides.append((name, ids, chain(name, ids)))
        count = max(1, ceil(max(s[2][1][-1] for s in sides) / 0.5))
        pair = []
        for side, (name, ids, (vertices, lengths, forward)) in zip(('a', 'b'), sides):
            points = []
            for j in range(count + 1):
                distance = lengths[-1] * j / count
                k = min(bisect_right(lengths, distance)-1, len(ids)-2)
                fraction = (distance-lengths[k]) / (lengths[k+1]-lengths[k])
                points.append([vertices[ids[k]][d] + fraction*(vertices[ids[k+1]][d]-vertices[ids[k]][d]) for d in range(2)])
            if not forward:
                points.reverse()
            start, end = (ids[0], ids[-1]) if forward else (ids[-1], ids[0])
            replacements[(name, start)] = (end, points, index, side, forward)
            pair.append((name, side, forward))
        plan.append((count, pair))

    edge_mapping = {}
    for name, panel in panels.items():
        original = panel['vertices']
        start = next((i for n, i in replacements if n == name), 0)
        current, vertices = start, []
        visited = set()
        while current not in visited:
            visited.add(current)
            item = replacements.get((name, current))
            if item:
                end, points, index, side, forward = item
                for j, point in enumerate(points[:-1]):
                    segment = j if forward else len(points)-2-j
                    edge_mapping[(index, side, segment)] = len(vertices)
                    vertices.append(point)
                current = end
            else:
                if (name, current) in owned:
                    raise ValueError(f'Overlapping seam replacement: {name}:{current}')
                vertices.append(original[current])
                current = (current+1) % len(original)
        if current != start:
            raise ValueError(f'Invalid boundary loop: {name}')
        panel['vertices'] = vertices
        panel['edges'] = [{'endpoints': [i, (i+1) % len(vertices)]} for i in range(len(vertices))]

    result = []
    for index, (count, pair) in enumerate(plan):
        for j in range(count):
            refs = [{'panel': name, 'edge': edge_mapping[(index, side, j)]} for name, side, _ in pair]
            if pair[0][2] == pair[1][2]:
                refs.append('right_wrong')
            result.append(refs)
    return result
