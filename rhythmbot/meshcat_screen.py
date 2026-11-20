"""Put the same minimal Sekai RGB frame on the Meshcat screen as a texture.

The plane uses the camera compositor's pixel-to-world calibration. glTF embeds
both pixels and geometry, so static Meshcat exports remain self-contained.
"""
from __future__ import annotations
import base64
import io
import json
from pathlib import Path
import tempfile
import weakref
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from pydrake.all import LeafSystem, Mesh
from .game import FrameRenderer
from .sekai_screen import SekaiScreen


def _close_texture_pool(pool, directory):
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)
    directory.cleanup()


def screen_quad(cfg):
    L = cfg.layout
    r = FrameRenderer(cfg)
    y0 = L.far_y()
    y1 = min(L.screen_center()[1] + L.screen_size()[1] / 2,
             L.hit_y + (r.H - r.hit_row) / r.px_per_sec * L.scroll_speed)
    v1 = (r.hit_row + (y1 - L.hit_y) / L.scroll_speed * r.px_per_sec) / r.H
    # Slightly above the existing lane markings, avoiding z-fighting.
    z = L.screen_z + .0015
    positions = np.array([[-L.half_width, y0, z], [L.half_width, y0, z],
                          [L.half_width, y1, z], [-L.half_width, y1, z]], dtype='<f4')
    uv = np.array([[0, 0], [1, 0], [1, v1], [0, v1]], dtype='<f4')
    return positions, uv


def write_screen_gltf(path, rgb, cfg):
    positions, uv = screen_quad(cfg)
    # glTF is Y-up; Drake converts its imported meshes back to Z-up.
    positions = positions[:, [0, 2, 1]].copy()
    positions[:, 2] *= -1
    indices = np.array([0, 1, 2, 0, 2, 3], dtype='<u2')
    chunks = [positions.tobytes(), uv.tobytes(), indices.tobytes()]
    binary = b''.join(chunks)
    png = io.BytesIO()
    Image.fromarray(rgb).save(png, format='PNG', compress_level=1)
    offsets = [0, len(chunks[0]), len(chunks[0]) + len(chunks[1])]
    gltf = {
        'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
        'nodes': [{'mesh': 0}],
        'buffers': [{'byteLength': len(binary), 'uri': 'data:application/octet-stream;base64,' + base64.b64encode(binary).decode()}],
        'bufferViews': [{'buffer': 0, 'byteOffset': off, 'byteLength': len(data)} for off, data in zip(offsets, chunks)],
        'accessors': [
            {'bufferView': 0, 'componentType': 5126, 'count': 4, 'type': 'VEC3', 'min': positions.min(0).tolist(), 'max': positions.max(0).tolist()},
            {'bufferView': 1, 'componentType': 5126, 'count': 4, 'type': 'VEC2'},
            {'bufferView': 2, 'componentType': 5123, 'count': 6, 'type': 'SCALAR'},
        ],
        'images': [{'uri': 'data:image/png;base64,' + base64.b64encode(png.getvalue()).decode()}],
        'samplers': [{'magFilter': 9729, 'minFilter': 9729, 'wrapS': 33071, 'wrapT': 33071}],
        'textures': [{'source': 0, 'sampler': 0}],
        'extensionsUsed': ['KHR_materials_unlit'],
        'materials': [{'doubleSided': True, 'extensions': {'KHR_materials_unlit': {}},
                       'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'metallicFactor': 0, 'roughnessFactor': 1}}],
        'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'TEXCOORD_0': 1}, 'indices': 2, 'material': 0}]}],
    }
    Path(path).write_text(json.dumps(gltf))


class MeshcatScreen(LeafSystem):
    def __init__(self, chart, cfg, meshcat):
        super().__init__()
        self.chart, self.cfg, self.meshcat = chart, cfg, meshcat
        self.renderer = SekaiScreen(cfg)
        self._directory = tempfile.TemporaryDirectory(prefix='rhythmbot-display-')
        self._path = Path(self._directory.name) / 'screen.gltf'
        self._pool = None
        self._pending = {}
        if cfg.camera.prefetch_noise:
            # Presentation only: prepare the next few deterministic highway
            # textures. Camera RGB still renders current robot occlusions and
            # current screen pixels synchronously, independently of this cache.
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='meshcat-texture')
            self._fill(0)
        self._cleanup = weakref.finalize(self, _close_texture_pool, self._pool, self._directory)
        meshcat.Delete('/highway')
        self.DeclarePeriodicPublishEvent(1 / cfg.visualizer_hz, 0., self._publish)
        self.DeclareForcedPublishEvent(self._publish)

    def _publish(self, context):
        t = context.get_time()
        tick = round(t * self.cfg.visualizer_hz)
        if self._pool is not None and abs(t - tick / self.cfg.visualizer_hz) < 1e-10:
            self._fill(tick)
            path = self._pending[tick].result()
        else:
            path = self._prepare(self._path, t)
        self.meshcat.SetObject('/sekai_screen', Mesh(str(path)))

    def _prepare(self, path, t):
        rgb = SekaiScreen(self.cfg).render(self.chart, t, scale=4)
        write_screen_gltf(path, rgb, self.cfg)
        return path

    def _fill(self, tick):
        for old in list(self._pending):
            if old < tick or old >= tick + 4:
                future = self._pending.pop(old)
                # Wait before removing a file that may still be written.
                if not future.cancel():
                    future.result().unlink(missing_ok=True)
        for k in range(tick, tick + 4):
            if k not in self._pending:
                path = Path(self._directory.name) / f'{k}.gltf'
                self._pending[k] = self._pool.submit(self._prepare, path, k / self.cfg.visualizer_hz)

    def close(self):
        self._cleanup()
