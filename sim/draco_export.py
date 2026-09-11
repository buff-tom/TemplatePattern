"""Export a mesh-only, meter-unit Draco GLB for DRACOLoader('/draco/')."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import struct


def export_glb(garment: Path, output: Path, body: Path | None = None):
    import numpy as np
    import trimesh
    try:
        import DracoPy
    except ImportError as exc:
        raise RuntimeError('Draco export requires DracoPy==1.7.0; install it in the running Python environment') from exc
    doc = {'asset': {'version': '2.0', 'generator': 'TemplatePattern_final'},
           'extensionsUsed': ['KHR_draco_mesh_compression'], 'extensionsRequired': ['KHR_draco_mesh_compression'],
           'scene': 0, 'scenes': [{'nodes': []}], 'nodes': [], 'meshes': [],
           'accessors': [], 'bufferViews': [], 'buffers': [],
           'extras': {'decoderPath': '/draco/', 'units': 'meter', 'appearance': 'neutral material; geometry-only export'},
           'materials': [{'doubleSided': True, 'pbrMetallicRoughness': {'baseColorFactor': [.7,.75,.8,1], 'metallicFactor': 0, 'roughnessFactor': 1}}]}
    binary = bytearray()
    for source, scale in ((garment, .01), (body, 1.)):
        if source is None:
            continue
        mesh = trimesh.load(str(source), force='mesh', process=False)
        points = np.asarray(mesh.vertices, dtype=np.float64) * scale
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        if not points.size or not faces.size or not np.isfinite(points).all():
            raise ValueError(f'Empty/nonfinite mesh: {source}')
        encoded = DracoPy.encode(points, faces, quantization_bits=16, compression_level=7, preserve_order=True)
        decoded = DracoPy.decode(encoded)
        attribute = next(a for a in decoded.attributes if a['attribute_type'] == 0)
        view = len(doc['bufferViews'])
        doc['bufferViews'].append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(encoded)})
        binary.extend(encoded)
        binary.extend(b'\0' * (-len(binary) % 4))
        accessor = len(doc['accessors'])
        doc['accessors'].extend([
            {'componentType': 5126, 'count': len(decoded.points), 'type': 'VEC3', 'min': decoded.points.min(axis=0).tolist(), 'max': decoded.points.max(axis=0).tolist()},
            {'componentType': 5125, 'count': int(decoded.faces.size), 'type': 'SCALAR'},
        ])
        index = len(doc['meshes'])
        doc['meshes'].append({'name': source.stem, 'primitives': [{'attributes': {'POSITION': accessor}, 'indices': accessor+1, 'mode': 4, 'material': 0,
            'extensions': {'KHR_draco_mesh_compression': {'bufferView': view, 'attributes': {'POSITION': attribute['unique_id']}}}}]})
        doc['nodes'].append({'mesh': index, 'name': source.stem})
        doc['scenes'][0]['nodes'].append(index)
    doc['buffers'] = [{'byteLength': len(binary)}]
    header = json.dumps(doc, separators=(',', ':')).encode()
    header += b' ' * (-len(header) % 4)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        stream.write(struct.pack('<4sII', b'glTF', 2, 12+8+len(header)+8+len(binary)))
        stream.write(struct.pack('<I4s', len(header), b'JSON')); stream.write(header)
        stream.write(struct.pack('<I4s', len(binary), b'BIN\0')); stream.write(binary)
    return {'glb': str(output), 'decoder_path': '/draco/', 'appearance': 'geometry_only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--garment', type=Path, required=True, help='GarmentCode OBJ in centimeters')
    parser.add_argument('--body', type=Path, help='SMPL OBJ in meters')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_glb(args.garment, args.output, args.body), indent=2))


if __name__ == '__main__':
    main()
