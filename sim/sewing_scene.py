"""Export posed panel edges, actual stitch endpoints and body as meter-unit OBJ."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def export_scene(stage2_dir, output_dir):
    root, out = Path(stage2_dir).resolve(), Path(output_dir).resolve()
    spec = json.loads((root/'input/garment_specification.json').read_text())
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a new output directory: {out}')
    body = (root/'body/body.obj').read_text().splitlines()
    panels = spec['pattern']['panels']
    units = float(spec['properties']['units_in_meter'])
    world = {}
    for name,p in panels.items():
        if any('curvature' in e for e in p['edges']):
            raise ValueError('Curved edges are not supported by this outline exporter')
        v = np.column_stack((np.asarray(p['vertices']),np.zeros(len(p['vertices']))))
        world[name]=(Rotation.from_euler('xyz',p['rotation'],degrees=True).apply(v)+p['translation'])/units
    out.mkdir(parents=True,exist_ok=True)
    lines=['mtllib sewing_scene.mtl','o body','usemtl body']
    # Keep source OBJ vertex/face indices intact; emitted SMPL OBJ has no UVs.
    lines += [line for line in body if line.startswith(('v ','vn ','vt ','f '))]
    offset=sum(line.startswith('v ') for line in body)
    def segment(name,a,b,material):
        nonlocal offset
        lines.extend([f'o {name}',f'usemtl {material}',
                      'v '+' '.join(map(str,a)),'v '+' '.join(map(str,b)),f'l {offset+1} {offset+2}'])
        offset+=2
    for name,p in panels.items():
        for i,e in enumerate(p['edges']):
            a,b=world[name][e['endpoints']]
            segment(f'panel_{name}_edge_{i}',a,b,'outline')
    rows=[]
    for i,s in enumerate(spec['pattern']['stitches'],1):
        ends=[]
        for ref in s[:2]:
            ends.append(world[ref['panel']][panels[ref['panel']]['edges'][ref['edge']]['endpoints']])
        same=len(s)>2 and s[2]=='right_wrong'
        if not same:ends[1]=ends[1][::-1]
        for j,(a,b) in enumerate(zip(*ends),1):segment(f'S{i:03d}_endpoint_{j}',a,b,'stitch')
        rows.append([f'S{i:03d}',s[0]['panel'],s[0]['edge'],s[1]['panel'],s[1]['edge'],'same' if same else 'reversed'])
    (out/'sewing_scene.obj').write_text('\n'.join(lines)+'\n')
    (out/'sewing_scene.mtl').write_text('newmtl body\nKd 0.6 0.65 0.7\nd 0.35\nnewmtl outline\nKd 0.05 0.2 0.8\nnewmtl stitch\nKd 0.95 0.15 0.05\n')
    with (out/'sewing_pairs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['id','panel_a','edge_a','panel_b','edge_b','endpoint_order']);w.writerows(rows)
    (out/'README.txt').write_text('单位：米。body 为当前任务人体；panel_* 为初始摆放边界；Sxxx_endpoint_* 为实际缝合端点连线。\n这是初始装配关系，不是仿真受力或最终布料形状。OBJ 与 MTL 请放在同一目录。\nBlender 导入 Wavefront OBJ 后可在 Outliner 隐藏 body 或 S* 对象；线段可能需要启用线框/叠加层显示。材质透明效果依查看器而定。\n',encoding='utf-8')
    return {'obj':str(out/'sewing_scene.obj'),'stitches':len(rows),'units':'meters'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage2-dir',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    args=p.parse_args();print(json.dumps(export_scene(args.stage2_dir,args.output_dir),indent=2))
