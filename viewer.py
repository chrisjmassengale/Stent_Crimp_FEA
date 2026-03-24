"""
viewer.py — High-performance interactive 3D Stent FEA Viewer.

Rendering
---------
  Stars       : single glDrawArrays call (vertex arrays, bucketed by size)
  Stent mesh  : VBO — one GPU draw call per frame, updated only on frame change
  Real-time   : moving deployed/crimp diameter sliders rescales the mesh live
                via glBufferSubData — no simulation re-run needed
  Panel       : pygame Surface → GL texture, re-uploaded only when UI changes

Controls
--------
  Left-drag          : Rotate
  Scroll wheel       : Zoom
  Ctrl + Left-drag   : Pan
  ← / →             : Previous / next frame
  Right-click slider : Type an exact value + Enter
"""

import ctypes, os, sys, json, math, time, threading, subprocess, random, copy
from pathlib import Path

import numpy as np
import trimesh
import pygame
from pygame.locals import (
    QUIT, KEYDOWN, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, MOUSEWHEEL,
    DOUBLEBUF, OPENGL, K_ESCAPE, K_LEFT, K_RIGHT, K_RETURN, K_BACKSPACE,
    KMOD_CTRL,
)

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print("[viewer] PyOpenGL not found.  pip install PyOpenGL PyOpenGL_accelerate")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

WIN_W, WIN_H = 1440, 900
PANEL_W      = 430
VIEW_W       = WIN_W - PANEL_W
FPS          = 60

SETTINGS_FILE = Path(__file__).parent / "viewer_settings.json"

DEFAULT_SETTINGS = {
    "input":               "stent.STL",
    "crimp_diameter":      6.0,
    "deployed_diameter":  28.0,
    "output_dir":         "./frames",
    "n_crimp_steps":       50,
    "n_deploy_steps":      50,
    "transition_length":   0.45,
    "snap_speed":          3.0,
    "crown_dwell":         0.60,
    "expansion_exponent":  0.6,
    "tine_flare":          1.15,
}

# (key, label, min, max, step, fmt)
PARAM_SPECS = [
    ("crimp_diameter",      "Crimp Diameter (mm)",     1.0,  15.0,  0.1,  "{:.1f}"),
    ("deployed_diameter",   "Deployed Diameter (mm)", 10.0,  50.0,  0.5,  "{:.1f}"),
    ("n_crimp_steps",       "Crimp Steps",            10,   200,    1,    "{:.0f}"),
    ("n_deploy_steps",      "Deploy Steps",           10,   200,    1,    "{:.0f}"),
    ("transition_length",   "Transition Length",       0.1,   0.9,  0.01, "{:.2f}"),
    ("snap_speed",          "Snap Speed",              0.5,  10.0,  0.1,  "{:.1f}"),
    ("crown_dwell",         "Crown Dwell",             0.0,   1.0,  0.01, "{:.2f}"),
    ("expansion_exponent",  "Expansion Exponent",      0.1,   2.0,  0.05, "{:.2f}"),
    ("tine_flare",          "Tine Flare",              1.0,   1.5,  0.01, "{:.2f}"),
]

# Colours
C_BG         = (12,  12,  30)
C_PANEL      = (20,  20,  50)
C_PANEL_EDGE = (60,  65, 160)
C_TEXT       = (215, 220, 255)
C_TEXT_DIM   = (115, 120, 170)
C_ACCENT     = (100, 140, 255)
C_SLIDER_BG  = (38,  40,  88)
C_SLIDER_FG  = (85, 120, 245)
C_KNOB       = (195, 210, 255)
C_BTN_FRAME  = (50,  50, 110)
C_BTN_FRAMEH = (75,  75, 160)
C_BTN_REGEN  = (38, 115,  55)
C_BTN_RGENH  = (52, 155,  72)
C_BTN_SAVE   = (45,  70, 150)
C_BTN_SAVEH  = (60,  92, 195)
C_PREVIEW    = (255, 200,  45)
C_DIRTY_BG   = (32,  28,   8)
C_OK         = (80, 210, 100)
C_ERR        = (255,  70,  70)


# ══════════════════════════════════════════════════════════════════════════════
# Settings
# ══════════════════════════════════════════════════════════════════════════════

def load_settings():
    s = copy.deepcopy(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            stored = json.loads(SETTINGS_FILE.read_text())
            for k in s:
                if k in stored:
                    s[k] = stored[k]
        except Exception:
            pass
    return s

def save_settings(s):
    try:
        SETTINGS_FILE.write_text(json.dumps(s, indent=2))
    except Exception as e:
        print(f"[viewer] save failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MeshBuffer — VBO-backed stent renderer with real-time radial deform
# ══════════════════════════════════════════════════════════════════════════════

class MeshBuffer:
    """
    Stores one STL frame as an interleaved VBO:  [nx ny nz vx vy vz] × n_tris×3.

    Real-time preview: call set_xy_scale(s) to rescale XY radially.
    This updates only the needed bytes via glBufferSubData — no CPU re-upload.
    """

    def __init__(self):
        self._vbo      = None
        self._n_verts  = 0
        self._base     = None   # (N, 6) float32 — pristine data, never modified
        self.centre    = np.zeros(3, np.float32)
        self.fit_scale = 1.0
        self._xy_scale = 1.0

    # ── load ─────────────────────────────────────────────────────────────────

    def load(self, stl_path: str):
        """Load STL, build packed array, upload to GPU."""
        m   = trimesh.load(str(stl_path), process=False, force='mesh')
        v   = m.vertices.astype(np.float32)
        f   = m.faces.astype(np.int32)
        fn  = m.face_normals.astype(np.float32)

        # Expand to per-draw-vertex layout
        v_exp = v[f].reshape(-1, 3)                 # (N*3, 3)
        n_exp = np.repeat(fn, 3, axis=0)            # (N*3, 3)
        data  = np.ascontiguousarray(
            np.hstack([n_exp, v_exp]), dtype=np.float32)   # (N*3, 6)

        self._base    = data.copy()
        self._n_verts = len(data)
        self.centre   = ((v.min(0) + v.max(0)) * 0.5).astype(np.float32)
        ext           = float((v.max(0) - v.min(0)).max())
        self.fit_scale= 20.0 / max(ext, 1e-6)
        self._xy_scale= 1.0

        # Upload
        if self._vbo is None:
            self._vbo = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    # ── real-time XY rescale ─────────────────────────────────────────────────

    def set_xy_scale(self, s: float):
        """Rescale all vertex XY by s (radial diameter preview)."""
        if self._base is None or abs(s - self._xy_scale) < 1e-5:
            return
        self._xy_scale = s
        d = self._base.copy()
        d[:, 3] *= s    # vx
        d[:, 4] *= s    # vy
        # Approximate normal update: scale nx,ny then renormalise
        d[:, 0] *= s
        d[:, 1] *= s
        nlen = np.linalg.norm(d[:, :3], axis=1, keepdims=True)
        d[:, :3] /= np.maximum(nlen, 1e-7)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, d.nbytes, d)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    # ── draw ─────────────────────────────────────────────────────────────────

    def draw(self, preview: bool):
        if self._vbo is None or self._n_verts == 0:
            return
        stride = 24   # 6 floats × 4 bytes

        if preview:
            glColor4f(1.0, 0.87, 0.30, 0.92)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.5, 0.44, 0.15, 1.0))
        else:
            glColor4f(0.93, 0.94, 1.00, 1.00)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.45, 0.45, 0.55, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 46.0)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_VERTEX_ARRAY)
        glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(0))
        glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(12))

        # Filled
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glDrawArrays(GL_TRIANGLES, 0, self._n_verts)

        # Wireframe overlay
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(0.6)
        glColor4f(0.3, 0.35, 0.60, 0.18)
        glDrawArrays(GL_TRIANGLES, 0, self._n_verts)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDisable(GL_BLEND)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisable(GL_DEPTH_TEST)

    def free(self):
        if self._vbo is not None:
            glDeleteBuffers(1, [self._vbo])
            self._vbo = None


# ══════════════════════════════════════════════════════════════════════════════
# Frame cache
# ══════════════════════════════════════════════════════════════════════════════

class FrameCache:
    def __init__(self, frames_dir: str):
        self.dir   = Path(frames_dir)
        self.paths = []
        self.buf   = MeshBuffer()    # single GPU buffer, reloaded on frame change
        self._loaded_idx = -1
        self.refresh()

    def refresh(self):
        self.paths = sorted(self.dir.glob("frame_*.stl"))
        self._loaded_idx = -1

    @property
    def count(self):
        return len(self.paths)

    def ensure_loaded(self, idx: int):
        if not self.paths or not (0 <= idx < len(self.paths)):
            return False
        if idx == self._loaded_idx:
            return True
        try:
            self.buf.load(str(self.paths[idx]))
            self._loaded_idx = idx
            return True
        except Exception as e:
            print(f"[viewer] Frame {idx} load error: {e}")
            return False


# ══════════════════════════════════════════════════════════════════════════════
# Star field — all 320 stars in 3 glDrawArrays calls (bucketed by size)
# ══════════════════════════════════════════════════════════════════════════════

_STAR_COLS = [
    (1.00, 1.00, 1.00), (0.78, 0.82, 1.00), (1.00, 0.93, 0.76),
    (0.70, 0.78, 1.00), (1.00, 1.00, 0.84), (0.85, 0.70, 1.00),
]

class StarField:
    N      = 320
    NBIG   = 28     # large stars
    NMED   = 90     # medium stars
    # rest = small

    def __init__(self):
        self.t = 0.0
        # Pre-generate star data
        xs     = np.random.uniform(-1.05, 1.05, self.N).astype(np.float32)
        ys     = np.random.uniform(-1.05, 1.05, self.N).astype(np.float32)
        self._pos   = np.stack([xs, ys], axis=1)             # (N, 2)
        self._cols  = np.array([random.choice(_STAR_COLS) for _ in range(self.N)],
                                dtype=np.float32)             # (N, 3)
        self._phase = np.random.uniform(0, 6.28, self.N).astype(np.float32)
        self._freq  = np.random.uniform(0.4, 3.5, self.N).astype(np.float32)
        # Sort by brightness tier so we can split indices
        tiers = np.array([0]*self.NBIG + [1]*self.NMED +
                         [2]*(self.N - self.NBIG - self.NMED))
        np.random.shuffle(tiers)
        self._tiers = tiers
        # Comets
        self._comets = [_Comet() for _ in range(4)]

    def update(self, dt: float):
        self.t += dt
        for c in self._comets:
            c.update(dt)

    def draw(self):
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)   # additive blend = glow

        t  = self.t
        # Per-star alpha based on twinkle
        alpha = (0.45 + 0.55 * np.sin(self._phase + t * self._freq) ** 2
                 ).astype(np.float32)           # (N,)
        rgba  = np.hstack([self._cols,
                           alpha[:, None]]).astype(np.float32)  # (N, 4)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        for size, tier in [(2.8, 0), (1.7, 1), (1.0, 2)]:
            mask = self._tiers == tier
            if not mask.any():
                continue
            pos_t  = np.ascontiguousarray(self._pos[mask])
            rgba_t = np.ascontiguousarray(rgba[mask])
            glPointSize(size)
            glVertexPointer(2, GL_FLOAT, 0, pos_t)
            glColorPointer(4,  GL_FLOAT, 0, rgba_t)
            glDrawArrays(GL_POINTS, 0, len(pos_t))

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        # Comets (few, drawn with immediate mode)
        glEnable(GL_POINT_SMOOTH)
        for c in self._comets:
            if not c.active:
                continue
            fade = max(0.0, 1.0 - c.life / c.max_life)
            r, g, b = c.col
            for i in range(c.TRAIL):
                frac  = i / c.TRAIL
                alpha = (1.0 - frac) * fade * 0.9
                sz    = (1.0 - frac) * 3.6
                glColor4f(r, g, b, alpha)
                glPointSize(max(0.4, sz))
                glBegin(GL_POINTS)
                glVertex2f(c.x - c.vx * frac * 0.09,
                           c.y - c.vy * frac * 0.09)
                glEnd()
        glDisable(GL_POINT_SMOOTH)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_BLEND)


class _Comet:
    TRAIL  = 26
    _COLORS = [(1.,.4,.1),(.35,.55,1.),(.8,.28,1.),(.25,1.,.65),
               (1.,.9,.15),(1.,1.,1.),(1.,1.,1.)]

    def __init__(self):
        self.active = False
        self.timer  = 0.0
        self.wait   = random.uniform(2., 9.)
        self.x = self.y = self.vx = self.vy = 0.
        self.life = self.max_life = 0.
        self.col = (1., 1., 1.)

    def spawn(self):
        e = random.randint(0, 3)
        if   e == 0: self.x, self.y =  random.uniform(-1,1),  1.15
        elif e == 1: self.x, self.y =  random.uniform(-1,1), -1.15
        elif e == 2: self.x, self.y = -1.15, random.uniform(-1,1)
        else:        self.x, self.y =  1.15, random.uniform(-1,1)
        a = math.atan2(-self.y, -self.x) + random.uniform(-0.45, 0.45)
        s = random.uniform(0.35, 1.1)
        self.vx, self.vy = math.cos(a)*s, math.sin(a)*s
        self.life, self.max_life = 0., random.uniform(0.7, 1.8)
        self.col    = random.choice(self._COLORS)
        self.active = True

    def update(self, dt):
        self.timer += dt
        if not self.active:
            if self.timer >= self.wait:
                self.timer = 0.; self.wait = random.uniform(3., 12.)
                self.spawn()
            return
        self.life += dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        if abs(self.x) > 1.6 or abs(self.y) > 1.6 or self.life > self.max_life:
            self.active = False
            self.timer  = 0.; self.wait = random.uniform(2., 9.)


# ══════════════════════════════════════════════════════════════════════════════
# Camera
# ══════════════════════════════════════════════════════════════════════════════

class Camera:
    def __init__(self):
        self.rx, self.ry = 18., -25.
        self.zoom        = 1.0
        self.px, self.py = 0., 0.
        self._p, self._mode = None, None

    def press(self, x, y, ctrl):
        self._p    = (x, y)
        self._mode = 'pan' if ctrl else 'rot'

    def drag(self, x, y):
        if not self._p:
            return
        dx, dy = x - self._p[0], y - self._p[1]
        self._p = (x, y)
        if self._mode == 'rot':
            self.ry += dx * 0.40; self.rx += dy * 0.40
        else:
            self.px += dx * 0.025 / self.zoom
            self.py -= dy * 0.025 / self.zoom

    def release(self):     self._p = None
    def scroll(self, d):   self.zoom = max(.08, min(12., self.zoom * (1.12 if d>0 else .89)))

    def apply(self):
        glScalef(self.zoom, self.zoom, self.zoom)
        glTranslatef(self.px, self.py, 0.)
        glRotatef(self.rx, 1, 0, 0)
        glRotatef(self.ry, 0, 1, 0)


# ══════════════════════════════════════════════════════════════════════════════
# OpenGL helpers
# ══════════════════════════════════════════════════════════════════════════════

def setup_lighting():
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT0, GL_POSITION, (2., 3., 3.5, 0.))
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  (.88,.88,.92,1.))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (.55,.55,.65,1.))
    glLightfv(GL_LIGHT0, GL_AMBIENT,  (0., 0., 0., 1.))
    glLightfv(GL_LIGHT1, GL_POSITION, (-1.5,-1.5,1.,0.))
    glLightfv(GL_LIGHT1, GL_DIFFUSE,  (.28,.28,.34,1.))
    glLightfv(GL_LIGHT1, GL_AMBIENT,  (.10,.10,.13,1.))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glShadeModel(GL_FLAT)


def surface_to_tex(surf: pygame.Surface) -> int:
    data = pygame.image.tostring(surf, "RGBA", True)
    w, h = surf.get_size()
    t    = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, t)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    return t


def draw_panel_tex(tex, view_w, win_w, win_h):
    glViewport(0, 0, win_w, win_h)
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    glOrtho(0, win_w, 0, win_h, -1, 1)
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity()
    glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D); glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBindTexture(GL_TEXTURE_2D, tex)
    glColor4f(1, 1, 1, 1)
    x1, x2, y1, y2 = float(view_w), float(win_w), 0., float(win_h)
    glBegin(GL_QUADS)
    glTexCoord2f(0,0); glVertex2f(x1,y1)
    glTexCoord2f(1,0); glVertex2f(x2,y1)
    glTexCoord2f(1,1); glVertex2f(x2,y2)
    glTexCoord2f(0,1); glVertex2f(x1,y2)
    glEnd()
    glDisable(GL_TEXTURE_2D); glDisable(GL_BLEND)


# ══════════════════════════════════════════════════════════════════════════════
# Slider widget
# ══════════════════════════════════════════════════════════════════════════════

class Slider:
    BAR_H  = 8
    KNOB_R = 9
    LBL_H  = 17

    def __init__(self, key, label, lo, hi, step, fmt, rect):
        self.key, self.label  = key, label
        self.lo, self.hi, self.step = lo, hi, step
        self.fmt, self.rect   = fmt, rect
        self.dragging = False
        self.editing  = False
        self.edit_buf = ""

    @property
    def bar(self):
        r = self.rect
        y = r.y + self.LBL_H + (r.h - self.LBL_H - self.BAR_H) // 2
        return pygame.Rect(r.x, y, r.w, self.BAR_H)

    def val_x(self, v):
        br = self.bar
        return br.x + (v - self.lo) / (self.hi - self.lo) * br.w

    def x_val(self, x):
        br  = self.bar
        t   = max(0., min(1., (x - br.x) / br.w))
        raw = self.lo + t * (self.hi - self.lo)
        return max(self.lo, min(self.hi, round(raw / self.step) * self.step))

    def knob_hit(self, mx, my, v):
        kx, ky = self.val_x(v), self.bar.centery
        return (mx-kx)**2 + (my-ky)**2 <= (self.KNOB_R+5)**2

    def bar_hit(self, mx, my):
        return self.bar.inflate(0, 12).collidepoint(mx, my)


# ══════════════════════════════════════════════════════════════════════════════
# UI Panel — dirty-flag rendering (re-draws Surface only when state changes)
# ══════════════════════════════════════════════════════════════════════════════

class UIPanel:
    PAD   = 14
    ROW_H = 50
    BTN_H = 40

    def __init__(self, w, h, settings, frame_cache):
        self.w, self.h        = w, h
        self.committed        = copy.deepcopy(settings)
        self.preview          = copy.deepcopy(settings)
        self.frame_cache      = frame_cache
        self.cur_frame        = 0
        self.dirty            = False   # preview ≠ committed
        self.sim_running      = False
        self.status_msg       = ""
        self.status_col       = C_TEXT_DIM
        self.hover            = None
        self.needs_redraw     = True    # panel Surface needs re-render
        self.sliders: list    = []
        self._fnt_lg = self._fnt_md = self._fnt_sm = None
        self._layout_ready    = False

    def init_fonts_and_layout(self):
        try:
            self._fnt_lg = pygame.font.SysFont("Segoe UI", 17, bold=True)
            self._fnt_md = pygame.font.SysFont("Segoe UI", 14)
            self._fnt_sm = pygame.font.SysFont("Segoe UI", 13)
        except Exception:
            self._fnt_lg = pygame.font.SysFont(None, 19, bold=True)
            self._fnt_md = pygame.font.SysFont(None, 16)
            self._fnt_sm = pygame.font.SysFont(None, 14)

        PAD, ROW_H, BTN_H = self.PAD, self.ROW_H, self.BTN_H
        w = self.w - 2*PAD
        y = PAD

        self._title_y  = y;  y += 22
        self._banner_y = y;  y += 18 + 8
        nb = 44
        self._prev_btn = pygame.Rect(PAD,          y, nb, BTN_H)
        self._next_btn = pygame.Rect(PAD+nb+4,     y, nb, BTN_H)
        self._frame_lr = pygame.Rect(PAD+nb*2+10,  y, w-nb*2-10, BTN_H)
        y += BTN_H + 10
        self._sep1 = y;  y += 12
        self._ph_y = y;  y += 22

        self.sliders = []
        for key, label, lo, hi, step, fmt in PARAM_SPECS:
            rect = pygame.Rect(PAD, y, w, ROW_H)
            self.sliders.append(Slider(key, label, lo, hi, step, fmt, rect))
            y += ROW_H + 6

        self._sep2 = y;  y += 12
        half = (w-6)//2
        self._regen_btn = pygame.Rect(PAD,        y, half, BTN_H)
        self._save_btn  = pygame.Rect(PAD+half+6, y, half, BTN_H)
        y += BTN_H + 8
        self._status_y  = y
        self._layout_ready = True

    # ── render ────────────────────────────────────────────────────────────────

    def render(self) -> pygame.Surface:
        surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        bg   = (*C_DIRTY_BG, 240) if self.dirty else (*C_PANEL, 245)
        surf.fill(bg)
        pygame.draw.line(surf, C_PANEL_EDGE, (0,0), (0,self.h), 2)

        T = self._txt

        T(surf, self._fnt_lg, "Stent FEA Viewer", C_ACCENT, self.PAD, self._title_y)
        if self.dirty:
            T(surf, self._fnt_sm, "  PREVIEW — uncommitted changes", C_PREVIEW,
              self.PAD, self._banner_y)
        else:
            T(surf, self._fnt_sm, "  Settings committed", C_TEXT_DIM,
              self.PAD, self._banner_y)

        pygame.draw.line(surf, C_PANEL_EDGE,
                         (self.PAD, self._sep1), (self.w-self.PAD, self._sep1), 1)

        n = max(1, self.frame_cache.count)
        self._btn(surf, self._prev_btn, "<", C_BTN_FRAME, C_BTN_FRAMEH, "prev")
        self._btn(surf, self._next_btn, ">", C_BTN_FRAME, C_BTN_FRAMEH, "next")
        lbl = self._fnt_md.render(f"Frame  {self.cur_frame+1}  /  {n}", True, C_TEXT)
        surf.blit(lbl, (self._frame_lr.x+6,
                        self._frame_lr.y + (self.BTN_H - lbl.get_height())//2))

        T(surf, self._fnt_md, "Simulation Parameters", C_TEXT, self.PAD, self._ph_y)

        for sl in self.sliders:
            self._draw_slider(surf, sl, self.preview.get(sl.key, sl.lo))

        pygame.draw.line(surf, C_PANEL_EDGE,
                         (self.PAD, self._sep2), (self.w-self.PAD, self._sep2), 1)

        rl = "Regenerating..." if self.sim_running else "Regenerate"
        self._btn(surf, self._regen_btn, rl,  C_BTN_REGEN, C_BTN_RGENH,
                  "regen", self.sim_running)
        self._btn(surf, self._save_btn, "Save Settings", C_BTN_SAVE, C_BTN_SAVEH, "save")

        if self.status_msg:
            T(surf, self._fnt_sm, self.status_msg, self.status_col,
              self.PAD, self._status_y)

        self.needs_redraw = False
        return surf

    def _txt(self, s, f, t, c, x, y):
        s.blit(f.render(t, True, c), (x, y))

    def _btn(self, surf, rect, label, col, colh, name, disabled=False):
        c = tuple(max(0,v-25) for v in col) if disabled \
            else (colh if self.hover==name else col)
        pygame.draw.rect(surf, c, rect, border_radius=6)
        pygame.draw.rect(surf, C_PANEL_EDGE, rect, 1, border_radius=6)
        t = self._fnt_sm.render(label, True, C_TEXT_DIM if disabled else C_TEXT)
        surf.blit(t, (rect.x+(rect.w-t.get_width())//2,
                      rect.y+(rect.h-t.get_height())//2))

    def _draw_slider(self, surf, sl, val):
        r, br = sl.rect, sl.bar
        self._txt(surf, self._fnt_sm, sl.label, C_TEXT_DIM, r.x, r.y)
        if sl.editing:
            vs, vc = sl.edit_buf+"|", C_PREVIEW
        else:
            vs, vc = sl.fmt.format(val), C_ACCENT
        vt = self._fnt_sm.render(vs, True, vc)
        surf.blit(vt, (r.right-vt.get_width(), r.y))

        pygame.draw.rect(surf, C_SLIDER_BG, br, border_radius=4)
        t  = (val-sl.lo)/(sl.hi-sl.lo)
        fw = max(0, int(t*br.w))
        if fw:
            pygame.draw.rect(surf, C_SLIDER_FG,
                             pygame.Rect(br.x, br.y, fw, br.h), border_radius=4)
        kx, ky = int(sl.val_x(val)), br.centery
        pygame.draw.circle(surf, C_KNOB,   (kx, ky), sl.KNOB_R)
        pygame.draw.circle(surf, C_ACCENT, (kx, ky), sl.KNOB_R, 2)

    # ── events ────────────────────────────────────────────────────────────────

    def handle(self, event, offset_x: int) -> bool:
        if not self._layout_ready:
            return False

        lx = lambda pos: (pos[0]-offset_x, pos[1])

        if event.type == MOUSEMOTION:
            mx, my = lx(event.pos)
            old_h  = self.hover
            self.hover = self._hittest(mx, my)
            if old_h != self.hover:
                self.needs_redraw = True
            for sl in self.sliders:
                if sl.dragging:
                    new_v = sl.x_val(mx)
                    if new_v != self.preview.get(sl.key):
                        self.preview[sl.key] = new_v
                        self.dirty = True
                        self.needs_redraw = True
            return False

        if event.type == MOUSEBUTTONDOWN:
            mx, my = lx(event.pos)

            if event.button == 3:   # right-click → text edit
                for sl in self.sliders:
                    if sl.rect.collidepoint(mx, my):
                        sl.editing  = True
                        sl.edit_buf = sl.fmt.format(self.preview.get(sl.key, sl.lo))
                        self.needs_redraw = True
                        return True

            if event.button == 1:
                for sl in self.sliders:
                    v = self.preview.get(sl.key, sl.lo)
                    if sl.knob_hit(mx, my, v) or sl.bar_hit(mx, my):
                        sl.dragging = True
                        new_v = sl.x_val(mx)
                        if new_v != v:
                            self.preview[sl.key] = new_v
                            self.dirty = True
                            self.needs_redraw = True
                        return True
                if self._prev_btn.collidepoint(mx, my):
                    self.cur_frame = max(0, self.cur_frame-1)
                    self.needs_redraw = True;  return True
                if self._next_btn.collidepoint(mx, my):
                    self.cur_frame = min(self.frame_cache.count-1, self.cur_frame+1)
                    self.needs_redraw = True;  return True
                if self._regen_btn.collidepoint(mx, my) and not self.sim_running:
                    self._start_regen();        return True
                if self._save_btn.collidepoint(mx, my):
                    save_settings(self.committed)
                    self.status_msg = "Settings saved."
                    self.status_col = C_OK
                    self.needs_redraw = True;  return True
            return False

        if event.type == MOUSEBUTTONUP and event.button == 1:
            for sl in self.sliders:
                sl.dragging = False
            return False

        if event.type == KEYDOWN:
            for sl in self.sliders:
                if sl.editing:
                    if event.key == K_RETURN:
                        try:
                            v = float(sl.edit_buf)
                            self.preview[sl.key] = max(sl.lo, min(sl.hi, v))
                            self.dirty = True
                        except ValueError:
                            pass
                        sl.editing = False
                    elif event.key == K_BACKSPACE:
                        sl.edit_buf = sl.edit_buf[:-1]
                    elif event.key == K_ESCAPE:
                        sl.editing = False
                    elif event.unicode in "0123456789.-":
                        sl.edit_buf += event.unicode
                    self.needs_redraw = True
                    return True
        return False

    def _hittest(self, mx, my):
        for n, r in [("prev",self._prev_btn),("next",self._next_btn),
                     ("regen",self._regen_btn),("save",self._save_btn)]:
            if r.collidepoint(mx, my):
                return n
        return None

    # ── simulation ────────────────────────────────────────────────────────────

    def _start_regen(self):
        self.committed   = copy.deepcopy(self.preview)
        self.dirty       = False
        self.sim_running = True
        self.status_msg  = "Simulation running..."
        self.status_col  = C_ACCENT
        self.needs_redraw = True
        threading.Thread(target=self._regen_worker, daemon=True).start()

    def _regen_worker(self):
        s    = self.committed
        base = Path(__file__).parent
        cmd  = [
            sys.executable, str(base/"simulate.py"),
            "--input",             str(base/s["input"]),
            "--crimp-diameter",    str(s["crimp_diameter"]),
            "--deployed-diameter", str(s["deployed_diameter"]),
            "--output-dir",        str(base/s["output_dir"]),
            "--n-crimp-steps",     str(int(s["n_crimp_steps"])),
            "--n-deploy-steps",    str(int(s["n_deploy_steps"])),
            "--transition-length", str(s["transition_length"]),
            "--snap-speed",        str(s["snap_speed"]),
            "--crown-dwell",       str(s["crown_dwell"]),
            "--expansion-exponent",str(s["expansion_exponent"]),
            "--tine-flare",        str(s["tine_flare"]),
            "--no-viewer",         # prevent simulate.py from spawning a new window
        ]
        try:
            r = subprocess.run(cmd, cwd=str(base),
                               capture_output=True, text=True, timeout=600)
            if r.returncode == 0:
                self.frame_cache.refresh()
                self.cur_frame    = 0
                self.status_msg   = f"Done — {self.frame_cache.count} frames."
                self.status_col   = C_OK
            else:
                self.status_msg = "Simulation failed — see console."
                self.status_col = C_ERR
                print(r.stderr[-2000:])
        except Exception as e:
            self.status_msg = f"Error: {e}"
            self.status_col = C_ERR
        finally:
            self.sim_running  = False
            self.needs_redraw = True

    # ── diameter preview scale ────────────────────────────────────────────────

    @property
    def preview_xy_scale(self) -> float:
        if not self.dirty:
            return 1.0
        cd = self.committed.get("deployed_diameter", 28.)
        pd = self.preview.get("deployed_diameter", 28.)
        return pd / cd if cd > 0 else 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(frames_dir: str = "./frames"):
    pygame.init()
    pygame.display.set_caption("Stent FEA Viewer")
    pygame.display.set_mode((WIN_W, WIN_H), DOUBLEBUF | OPENGL)

    glClearColor(*[c/255. for c in C_BG], 1.)
    glEnable(GL_NORMALIZE)
    setup_lighting()

    renderer = glGetString(GL_RENDERER).decode()
    print(f"[viewer] GPU: {renderer}")

    settings    = load_settings()
    frame_cache = FrameCache(frames_dir)
    starfield   = StarField()
    camera      = Camera()
    panel       = UIPanel(PANEL_W, WIN_H, settings, frame_cache)
    panel.init_fonts_and_layout()

    panel_tex   = None
    clock       = pygame.time.Clock()
    t_prev      = time.time()
    _loaded_f   = -1    # track which frame is in the VBO

    while True:
        now   = time.time()
        dt    = min(now - t_prev, 0.05)
        t_prev = now
        ctrl  = bool(pygame.key.get_mods() & KMOD_CTRL)

        # ── events ────────────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == QUIT:
                _quit(panel_tex, frame_cache); return
            if ev.type == KEYDOWN:
                if ev.key == K_ESCAPE:
                    _quit(panel_tex, frame_cache); return
                if ev.key == K_LEFT:
                    panel.cur_frame = max(0, panel.cur_frame-1)
                    panel.needs_redraw = True
                if ev.key == K_RIGHT:
                    panel.cur_frame = min(frame_cache.count-1, panel.cur_frame+1)
                    panel.needs_redraw = True
                panel.handle(ev, VIEW_W)
            elif ev.type == MOUSEBUTTONDOWN:
                if ev.pos[0] < VIEW_W:  camera.press(ev.pos[0], ev.pos[1], ctrl)
                else:                   panel.handle(ev, VIEW_W)
            elif ev.type == MOUSEBUTTONUP:
                camera.release(); panel.handle(ev, VIEW_W)
            elif ev.type == MOUSEMOTION:
                if ev.pos[0] < VIEW_W:  camera.drag(ev.pos[0], ev.pos[1])
                else:                   panel.handle(ev, VIEW_W)
            elif ev.type == MOUSEWHEEL:
                if pygame.mouse.get_pos()[0] < VIEW_W:
                    camera.scroll(ev.y)

        # ── update ────────────────────────────────────────────────────────────
        starfield.update(dt)

        # Load frame into VBO if frame index changed
        if panel.cur_frame != _loaded_f:
            frame_cache.ensure_loaded(panel.cur_frame)
            _loaded_f = panel.cur_frame
            panel.needs_redraw = True   # update frame counter text

        # Apply real-time XY scale for diameter preview
        frame_cache.buf.set_xy_scale(panel.preview_xy_scale)

        # ── render ────────────────────────────────────────────────────────────
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 1 ── Star field (orthographic overlay over 3-D viewport) ─────────────
        glViewport(0, 0, VIEW_W, WIN_H)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW);  glLoadIdentity()
        starfield.draw()

        # 2 ── 3-D stent ───────────────────────────────────────────────────────
        glViewport(0, 0, VIEW_W, WIN_H)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(42., VIEW_W/WIN_H, 0.1, 800.)
        glMatrixMode(GL_MODELVIEW);  glLoadIdentity()
        glTranslatef(0., 0., -55.)
        camera.apply()

        buf = frame_cache.buf
        if buf._vbo is not None:
            glScalef(buf.fit_scale, buf.fit_scale, buf.fit_scale)
            c = buf.centre
            glTranslatef(-float(c[0]), -float(c[1]), -float(c[2]))
            buf.draw(preview=panel.dirty)

        # 3 ── UI panel (re-upload texture only when dirty) ────────────────────
        if panel.needs_redraw:
            surf = panel.render()
            if panel_tex is not None:
                glDeleteTextures(1, [panel_tex])
            panel_tex = surface_to_tex(surf)

        if panel_tex is not None:
            draw_panel_tex(panel_tex, VIEW_W, WIN_W, WIN_H)

        pygame.display.flip()
        clock.tick(FPS)


def _quit(panel_tex, frame_cache):
    if panel_tex:
        try: glDeleteTextures(1, [panel_tex])
        except Exception: pass
    frame_cache.buf.free()
    pygame.quit()


if __name__ == "__main__":
    fd = sys.argv[1] if len(sys.argv) > 1 else "./frames"
    main(frames_dir=fd)
