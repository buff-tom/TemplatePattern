"""OpenGL compatibility and a small preflight before expensive simulation."""
import os


def configure_opengl():
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    from OpenGL.arrays import arraydatatype
    from OpenGL.arrays.ctypesparameters import CtypesParameterHandler
    # Python 3.12's byref type is not found by PyOpenGL 3.1.0's string registry.
    arraydatatype.GLOBAL_REGISTRY.register(CtypesParameterHandler(), CtypesParameterHandler.HANDLED_TYPES)


def preflight():
    configure_opengl()
    import numpy as np
    import pyrender
    import trimesh
    from PIL import Image
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)), image=Image.new('RGB', (2, 2), 'white'))
    scene = pyrender.Scene(ambient_light=[1., 1., 1.])
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    pose = np.eye(4)
    pose[2, 3] = 3
    scene.add(pyrender.PerspectiveCamera(yfov=0.8), pose=pose)
    renderer = None
    try:
        renderer = pyrender.OffscreenRenderer(16, 16)
        color, _ = renderer.render(scene)
        if color.shape[:2] != (16, 16):
            raise RuntimeError('Unexpected offscreen render shape')
    finally:
        if renderer is not None:
            renderer.delete()


if __name__ == '__main__':
    preflight()
    print('EGL textured render preflight passed')
