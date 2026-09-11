"""Read-only sewing graph visualization: offline HTML, SVG and CSV."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from html import escape
import json
from math import ceil, dist
from pathlib import Path

PALETTE = ['#2563eb','#dc2626','#059669','#9333ea','#d97706','#0891b2','#db2777','#4f46e5','#65a30d','#a16207','#0f766e','#7c3aed']


def visualize(source: Path, output: Path):
    source, output = source.resolve(), output.resolve()
    data = json.loads(source.read_text())['pattern']
    panels = data['panels']
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new visualization directory: {output}')
    boxes = {n:(min(v[0] for v in p['vertices']),min(v[1] for v in p['vertices']),max(v[0] for v in p['vertices']),max(v[1] for v in p['vertices'])) for n,p in panels.items()}
    scale = min(240/max(b[2]-b[0] for b in boxes.values()),250/max(b[3]-b[1] for b in boxes.values()))
    mapped, base = {}, []
    for i,(name,panel) in enumerate(panels.items()):
        if any('curvature' in e for e in panel['edges']):
            raise ValueError('Curved edges require curve sampling; refusing to draw misleading straight chords')
        x0,y0,x1,y1=boxes[name]
        cx,cy=150+300*(i%4),165+330*(i//4)
        points=[[cx+(x-(x0+x1)/2)*scale,cy-(y-(y0+y1)/2)*scale] for x,y in panel['vertices']]
        mapped[name]=points
        base.append(f'<text x="{cx}" y="{cy-140}" text-anchor="middle" font-size="13">{escape(name)}</text>')
        for edge in panel['edges']:
            a,b=[points[j] for j in edge['endpoints']]
            base.append(f'<path d="M {a[0]} {a[1]} L {b[0]} {b[1]}" fill="none" stroke="#94a3b8" stroke-width="1"/>')
    groups, rows, overlays = {}, [], []
    for i, stitch in enumerate(data['stitches']):
        a,b=stitch[:2]
        pair=tuple(sorted((a['panel'],b['panel'])))
        if pair not in groups:groups[pair]=len(groups)
        g=groups[pair];color=PALETTE[g%len(PALETTE)]
        sid=f'S{i+1:03d}'
        same=len(stitch)>2 and stitch[2]=='right_wrong'
        coords=[];lengths=[]
        for ref in (a,b):
            p=panels[ref['panel']];ids=p['edges'][ref['edge']]['endpoints']
            coords.append([mapped[ref['panel']][j] for j in ids])
            lengths.append(dist(*[p['vertices'][j] for j in ids]))
        first,second=coords
        if not same:second=list(reversed(second))
        overlay=[f'<g class="seam" data-id="{sid}" data-group="{g}">']
        for ref,points in zip((a,b),coords):
            p,q=points
            overlay.append(f'<path class="edge" d="M {p[0]} {p[1]} L {q[0]} {q[1]}" stroke="{color}" stroke-width="3" fill="none"><title>{sid}: {escape(ref["panel"])} edge {ref["edge"]}</title></path>')
        for number,(p,q) in enumerate(zip(first,second),1):
            overlay.append(f'<path class="connector" d="M {p[0]} {p[1]} L {q[0]} {q[1]}" stroke="{color}" stroke-width="1.2" stroke-dasharray="5 4" opacity=".65"/>')
            for x,y in (p,q):overlay.append(f'<text class="endpoint" x="{x+3}" y="{y-3}" fill="{color}" font-size="12">{number}</text>')
        overlay.append('</g>');overlays.extend(overlay)
        rows.append({'id':sid,'group':g,'group_label':f'G{g+1:02d}', 'panel_a':a['panel'],'edge_a':a['edge'], 'panel_b':b['panel'],'edge_b':b['edge'], 'flag':'right_wrong' if same else 'default_reverse', 'endpoint_pairing':'a0-b0; a1-b1' if same else 'a0-b1; a1-b0', 'length_a_cm':round(lengths[0],5),'length_b_cm':round(lengths[1],5)})
    height=ceil(len(panels)/4)*330
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {height}" id="drawing"><rect width="1200" height="{height}" fill="white"/>'+''.join(base)+''.join(overlays)+'</svg>'
    options='<option value="all">全部缝合（总览）</option>'+''.join(f'<option value="{g}">G{g+1:02d}: {escape(a)} ↔ {escape(b)}</option>' for (a,b),g in groups.items())
    table=''.join(f'<tr data-id="{r["id"]}" data-group="{r["group"]}"><td>{r["id"]}</td><td>{r["group_label"]}</td><td>{escape(r["panel_a"])}:e{r["edge_a"]}</td><td>{escape(r["panel_b"])}:e{r["edge_b"]}</td><td>{r["endpoint_pairing"]}</td><td>{r["length_a_cm"]} / {r["length_b_cm"]}</td></tr>' for r in rows)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>缝合关系人工检查</title>
<style>body{font:14px system-ui;margin:20px;color:#172033}header{position:sticky;top:0;background:white;padding:10px;z-index:2;border-bottom:1px solid #ddd}main{display:grid;grid-template-columns:60% 40%;gap:12px}svg{width:100%;height:auto}table{border-collapse:collapse;font-size:11px;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}tr{cursor:pointer}tr.selected{background:#fef3c7}tr:hover{background:#e0f2fe}.connector,.endpoint{display:none}.seam{cursor:pointer}button,select{padding:7px}aside{max-height:85vh;overflow:auto}@media(max-width:900px){main{display:block}}</style>
<header><h2>原始缝合关系 · 人工检查</h2><p>四列平铺显示各片局部二维坐标（统一比例），不是三维摆放图。边索引从 0 开始；S 编号按 stitches 原顺序从 1 开始。</p>
<label>纸片配对 <select id="group">OPTIONS</select></label> <button id="reset">清除选中</button>
<p id="status">点击表格的一条 S 缝合：高亮两条边，并用虚线连接实际配对的 1/2 端点。相同端点数字应缝合。</p>
<p>颜色仅作辅助；请以 S 编号、纸片名和边编号为准。边长不同不自动判为错误（可能存在吃势）。原缝合数据不会被修改。</p></header>
<main><section>DRAWING</section><aside><table><thead><tr><th>S</th><th>组</th><th>A</th><th>B</th><th>端点配对</th><th>边长 A/B cm</th></tr></thead><tbody>ROWS</tbody></table></aside></main>
<script>const seams=[...document.querySelectorAll('.seam')], rows=[...document.querySelectorAll('tbody tr')], group=document.querySelector('#group');
function filter(){seams.forEach(s=>{s.style.display=group.value==='all'||s.dataset.group===group.value?'':'none';s.style.opacity=1;s.querySelectorAll('.connector,.endpoint').forEach(e=>e.style.display='none')});rows.forEach(r=>{r.style.display=group.value==='all'||r.dataset.group===group.value?'':'none';r.classList.remove('selected')});}
function select(id){const row=rows.find(r=>r.dataset.id===id);group.value=row.dataset.group;filter();seams.forEach(s=>{s.style.opacity=s.dataset.id===id?1:.15;if(s.dataset.id===id)s.querySelectorAll('.connector,.endpoint').forEach(e=>e.style.display='block')});row.classList.add('selected');document.querySelector('#status').textContent='当前选中 '+row.innerText;}
group.onchange=filter;document.querySelector('#reset').onclick=filter;rows.forEach(r=>r.onclick=()=>select(r.dataset.id));seams.forEach(s=>s.onclick=()=>select(s.dataset.id));filter();</script></html>'''
    html=html.replace('OPTIONS',options).replace('DRAWING',svg).replace('ROWS',table)
    output.mkdir(parents=True,exist_ok=True)
    (output/'sewing_map.html').write_text(html,encoding='utf-8')
    static=svg.replace('<rect','<style>.connector,.endpoint{display:none}</style><rect',1)
    (output/'sewing_overview.svg').write_text(static,encoding='utf-8')
    with (output/'sewing_pairs.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'source':str(source),'panels':len(panels),'stitches':len(rows),'groups':len(groups),'units':'cm','read_only':True}
    (output/'visualization_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'html':str(output/'sewing_map.html'),**summary}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(visualize(args.spec,args.output_dir),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
