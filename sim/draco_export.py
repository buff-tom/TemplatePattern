"""Export textured garments and a black body in a self-contained Draco GLB."""
from __future__ import annotations
import argparse
import json
from io import BytesIO
from pathlib import Path
import struct


def _buffer_view(doc, binary, data):
    index = len(doc['bufferViews'])
    doc['bufferViews'].append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(data)})
    binary.extend(data)
    binary.extend(b'\0' * (-len(binary) % 4))
    return index


def _garment_material(mesh, doc, binary):
    import numpy as np
    from PIL import Image

    # A contrasting fallback remains usable when only a geometry OBJ survives.
    material = {'name': 'garment_colored', 'doubleSided': True, 'alphaMode': 'OPAQUE',
                'pbrMetallicRoughness': {'baseColorFactor': [.25, .55, .8, 1.],
                                         'metallicFactor': 0., 'roughnessFactor': .85}}
    source_material = getattr(mesh.visual, 'material', None)
    image = getattr(source_material, 'image', None)
    if image is None:
        image = getattr(source_material, 'baseColorTexture', None)
    uv = getattr(mesh.visual, 'uv', None)
    if image is None or uv is None:
        return material, None
    uv = np.asarray(uv, dtype=np.float64).copy()
    if uv.shape != (len(mesh.vertices), 2) or not np.isfinite(uv).all():
        raise ValueError('Garment texture requires finite UV coordinates for every vertex')
    # OBJ/trimesh uses a lower-left origin; glTF uses an upper-left origin.
    uv[:, 1] = 1. - uv[:, 1]
    # Keep opaque fabric like the simulation renderer; atlas transparency must
    # not turn the shirt into transparent geometry in a glTF viewer.
    rgba = image.convert('RGBA')
    opaque = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    opaque.alpha_composite(rgba)
    stream = BytesIO()
    opaque.convert('RGB').save(stream, format='PNG')
    view = _buffer_view(doc, binary, stream.getvalue())
    doc['images'] = [{'name': 'garment_texture', 'bufferView': view, 'mimeType': 'image/png'}]
    doc['samplers'] = [{'magFilter': 9729, 'minFilter': 9729, 'wrapS': 10497, 'wrapT': 10497}]
    doc['textures'] = [{'source': 0, 'sampler': 0}]
    material['pbrMetallicRoughness'].update({
        'baseColorFactor': [1., 1., 1., 1.],
        'baseColorTexture': {'index': 0, 'texCoord': 0},
    })
    return material, uv


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
           'extras': {'decoderPath': '/draco/', 'units': 'meter', 'appearance': 'colored garment and black body'},
           'materials': []}
    binary = bytearray()
    textured = False
    for role, source, scale in (('garment', garment, .01), ('body', body, 1.)):
        if source is None:
            continue
        mesh = trimesh.load(str(source), force='mesh', process=False)
        points = np.asarray(mesh.vertices, dtype=np.float64) * scale
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        if not points.size or not faces.size or not np.isfinite(points).all():
            raise ValueError(f'Empty/nonfinite mesh: {source}')
        uv = None
        if role == 'garment':
            material, uv = _garment_material(mesh, doc, binary)
            textured = uv is not None
        else:
            material = {'name': 'body_black', 'alphaMode': 'OPAQUE',
                        'pbrMetallicRoughness': {'baseColorFactor': [0., 0., 0., 1.],
                                               'metallicFactor': .658, 'roughnessFactor': .5}}
        material_index = len(doc['materials'])
        doc['materials'].append(material)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        if normals.shape != points.shape or not np.isfinite(normals).all():
            raise ValueError(f'Invalid mesh normals: {source}')
        encoded = DracoPy.encode(points, faces, quantization_bits=16, compression_level=7,
                                 preserve_order=True, normals=normals, tex_coord=uv)
        decoded = DracoPy.decode(encoded)
        view = _buffer_view(doc, binary, encoded)
        attributes, draco_attributes = {}, {}
        semantics = [('POSITION', 0, decoded.points, 'VEC3'), ('NORMAL', 1, decoded.normals, 'VEC3')]
        if uv is not None:
            semantics.append(('TEXCOORD_0', 3, decoded.tex_coord, 'VEC2'))
        for semantic, attribute_type, values, shape in semantics:
            attribute = next(a for a in decoded.attributes if a['attribute_type'] == attribute_type)
            attributes[semantic] = len(doc['accessors'])
            draco_attributes[semantic] = attribute['unique_id']
            accessor = {'componentType': 5126, 'count': len(values), 'type': shape}
            if semantic == 'POSITION':
                accessor.update({'min': values.min(axis=0).tolist(), 'max': values.max(axis=0).tolist()})
            doc['accessors'].append(accessor)
        indices = len(doc['accessors'])
        doc['accessors'].append({'componentType': 5125, 'count': int(decoded.faces.size), 'type': 'SCALAR'})
        index = len(doc['meshes'])
        doc['meshes'].append({'name': role, 'primitives': [{'attributes': attributes, 'indices': indices, 'mode': 4, 'material': material_index,
            'extensions': {'KHR_draco_mesh_compression': {'bufferView': view, 'attributes': draco_attributes}}}]})
        doc['nodes'].append({'mesh': index, 'name': role})
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
    return {'glb': str(output), 'decoder_path': '/draco/',
            'appearance': 'textured_garment_black_body' if textured else 'colored_garment_black_body',
            'garment_texture_embedded': textured}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--garment', type=Path, required=True, help='GarmentCode OBJ in centimeters')
    parser.add_argument('--body', type=Path, help='SMPL OBJ in meters')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_glb(args.garment, args.output, args.body), indent=2))


if __name__ == '__main__':
    main()
