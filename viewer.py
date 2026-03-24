"""
viewer.py — Interactive 3D Stent FEA Viewer.

Left pane  : Animated starfield + 3-D OpenGL stent model
Right pane : Control panel — frame navigator, simulation parameters,
             regenerate & save buttons.

Mouse controls (in 3-D viewport)
---------------------------------
  Left-drag          → Rotate
  Scroll wheel       → Zoom
  Ctrl + Left-drag   → Pan

Frame navigation
---------------------------------
  ← / →  or on-screen ◀ / ▶ buttons

Parameter editing
---------------------------------
  Drag slider OR right-click a slider value to type a number + Enter.
  Changes are immediately shown as a scaled preview (yellow tint).
  Click "Regenerate" to run the full simulation with the new values.
  Click "Save Settings" to persist across sessions.

Usage
---------------------------------
  python viewer.py [frames_dir]      (default: ./frames)
"""

import os, sys, json, math, time, threading, subprocess, random, copy
from pathlib import Path

import numpy as np
import trimesh
import pygame
from pygame.locals import (
    QUIT, KEYDOWN, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, MOUSEWHEEL,
    DOUBLEBUF, OPENGL, K_ESCAPE, K_LEFT, K_RIGHT, K_RETURN, K_BACKSPACE,
    K_ESCAPE, KMOD_CTRL,
)

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print("[viewer] PyOpenGL not found.  Install: pip install PyOpenGL PyOpenGL_accelerate")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# Layout & colour constants
# ══════════════════════════════════════════════════════════════════════════════

WIN_W    = 1440
WIN_H    = 900
PANEL_W  = 430
VIEW_W   = WIN_W - PANEL_W
TARGET_FPS = 60

SETTINGS_FILE = Path(__file__).parent / "viewer_settings.json"

# UI colours (R, G, B) — used both for pygame and normalised for GL
C_BG          = (12,  12,  30)
C_PANEL       = (20,  20,  50)
C_PANEL_EDGE  = (60,  65, 160)
C_TEXT        = (215, 220, 255)
C_TEXT_DIM    = (120, 125, 175)
C_ACCENT      = (100, 140, 255)
C_SLIDER_BG   = (38,  40,  88)
C_SLIDER_FG   = (85, 120, 245)
C_KNOB        = (195, 210, 255)
C_BTN_FRAME   = (50,  50, 110)
C_BTN_FRAME_H = (75,  75, 160)
C_BTN_REGEN   = (38, 115,  55)
C_BTN_REGEN_H = (52, 155,  72)
C_BTN_SAVE    = (45,  70, 150)
C_BTN_SAVE_H  = (60,  92, 195)
C_PREVIEW     = (255, 200,  45)
C_DIRTY_BG    = (35,  30,  10)
C_OK          = (80, 210, 100)
C_ERR         = (255,  70,  70)


# ══════════════════════════════════════════════════════════════════════════════
# Simulation parameter specifications
# ══════════════════════════════════════════════════════════════════════════════

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

# (key, label, min, max, step, format)
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


# ══════════════════════════════════════════════════════════════════════════════
# Settings persistence
# ══════════════════════════════════════════════════════════════════════════════

def load_settings() -> dict:
    s = copy.deepcopy(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                stored = json.load(f)
            for k in s:
                if k in stored:
                    s[k] = stored[k]
        except Exception as e:
            print(f"[viewer] Settings load failed: {e}")
    return s


def save_settings(s: dict):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(s, f, indent=2)
    except Exception as e:
        print(f"[viewer] Settings save failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Frame cache — lazy STL loader
# ══════════════════════════════════════════════════════════════════════════════

class FrameCache:
    MAX = 25   # frames kept in RAM

    def __init__(self, frames_dir: str):
        self.dir    = Path(frames_dir)
        self.paths: list = []
        self._cache: dict = {}    # index → (verts, faces, normals, centre, scale)
        self.refresh()

    def refresh(self):
        self.paths  = sorted(self.dir.glob("frame_*.stl"))
        self._cache = {}

    @property
    def count(self) -> int:
        return len(self.paths)

    def get(self, idx: int):
        """Return (vertices, faces, normals, centre, fit_scale) or Nones."""
        if not self.paths or not (0 <= idx < len(self.paths)):
            return None, None, None, None, None
        if idx not in self._cache:
            if len(self._cache) >= self.MAX:
                del self._cache[next(iter(self._cache))]
            try:
                m   = trimesh.load(str(self.paths[idx]), process=False, force='mesh')
                v   = m.vertices.astype(np.float32)
                f   = m.faces.astype(np.int32)
                fn  = m.face_normals.astype(np.float32)
                ctr = ((v.min(0) + v.max(0)) * 0.5).astype(np.float32)
                ext = (v.max(0) - v.min(0)).max()
                scl = float(20.0 / max(ext, 1e-6))   # fit to ~20-unit box
                self._cache[idx] = (v, f, fn, ctr, scl)
            except Exception as e:
                print(f"[viewer] Frame {idx} load error: {e}")
                return None, None, None, None, None
        return self._cache[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Star field — background particle system
# ══════════════════════════════════════════════════════════════════════════════

_STAR_PALETTES = [
    (1.00, 1.00, 1.00), (0.78, 0.82, 1.00), (1.00, 0.94, 0.78),
    (0.70, 0.78, 1.00), (1.00, 1.00, 0.86), (0.86, 0.72, 1.00),
]

class _Star:
    __slots__ = ("x", "y", "sz", "col", "phase", "freq", "depth")

    def __init__(self):
        self.x     = random.uniform(-1.05, 1.05)
        self.y     = random.uniform(-1.05, 1.05)
        self.sz    = random.uniform(0.6, 2.4)
        self.col   = random.choice(_STAR_PALETTES)
        self.phase = random.uniform(0, 6.28)
        self.freq  = random.uniform(0.4, 3.5)
        self.depth = random.uniform(0.3, 1.0)

    def alpha(self, t):
        return 0.45 + 0.55 * math.sin(self.phase + t * self.freq) ** 2


class _Comet:
    _COLORS = [
        (1.0, 0.40, 0.10),   # orange
        (0.35, 0.55, 1.00),  # blue
        (0.80, 0.28, 1.00),  # violet
        (0.25, 1.00, 0.65),  # teal
        (1.00, 0.90, 0.15),  # gold
        (1.00, 1.00, 1.00),  # white
        (1.00, 1.00, 1.00),  # white (higher chance)
    ]
    TRAIL = 28

    def __init__(self):
        self.active  = False
        self.timer   = 0.0
        self.wait    = random.uniform(2.0, 9.0)
        self.x = self.y = self.vx = self.vy = 0.0
        self.life = self.max_life = 0.0
        self.col = (1.0, 1.0, 1.0)

    def spawn(self):
        edge = random.randint(0, 3)
        if   edge == 0: self.x, self.y = random.uniform(-1,1),  1.15
        elif edge == 1: self.x, self.y = random.uniform(-1,1), -1.15
        elif edge == 2: self.x, self.y = -1.15, random.uniform(-1,1)
        else:           self.x, self.y =  1.15, random.uniform(-1,1)
        ang  = math.atan2(-self.y, -self.x) + random.uniform(-0.45, 0.45)
        spd  = random.uniform(0.35, 1.1)
        self.vx, self.vy = math.cos(ang)*spd, math.sin(ang)*spd
        self.life, self.max_life = 0.0, random.uniform(0.7, 1.8)
        self.col    = random.choice(self._COLORS)
        self.active = True

    def update(self, dt):
        self.timer += dt
        if not self.active:
            if self.timer >= self.wait:
                self.timer = 0.0
                self.wait  = random.uniform(3.0, 12.0)
                self.spawn()
            return
        self.life += dt
        self.x    += self.vx * dt
        self.y    += self.vy * dt
        if abs(self.x) > 1.6 or abs(self.y) > 1.6 or self.life > self.max_life:
            self.active = False
            self.timer  = 0.0
            self.wait   = random.uniform(2.0, 9.0)


class StarField:
    N = 320

    def __init__(self):
        self.stars  = [_Star() for _ in range(self.N)]
        self.comets = [_Comet() for _ in range(4)]
        self.t      = 0.0

    def update(self, dt):
        self.t += dt
        for c in self.comets:
            c.update(dt)

    def draw(self):
        """Draw in a [-1,+1]×[-1,+1] orthographic coordinate system."""
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)   # additive — looks great
        glEnable(GL_POINT_SMOOTH)

        t = self.t
        # ── stars ──
        for s in self.stars:
            a = s.alpha(t)
            r, g, b = s.col
            glColor4f(r, g, b, a)
            glPointSize(s.sz * (0.7 + 0.3 * a))
            glBegin(GL_POINTS)
            glVertex2f(s.x, s.y)
            glEnd()

        # ── comets ──
        for c in self.comets:
            if not c.active:
                continue
            fade = max(0.0, 1.0 - c.life / c.max_life)
            r, g, b = c.col
            for i in range(c.TRAIL):
                frac  = i / c.TRAIL
                alpha = (1.0 - frac) * fade * 0.92
                sz    = (1.0 - frac) * 3.8
                tx    = c.x - c.vx * frac * 0.09
                ty    = c.y - c.vy * frac * 0.09
                glColor4f(r, g, b, alpha)
                glPointSize(max(0.4, sz))
                glBegin(GL_POINTS)
                glVertex2f(tx, ty)
                glEnd()

        glDisable(GL_POINT_SMOOTH)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_BLEND)


# ══════════════════════════════════════════════════════════════════════════════
# Camera
# ══════════════════════════════════════════════════════════════════════════════

class Camera:
    def __init__(self):
        self.rot_x     =  18.0
        self.rot_y     = -25.0
        self.zoom      =   1.0
        self.pan_x     =   0.0
        self.pan_y     =   0.0
        self._start    =  None
        self._mode     =  None   # 'rotate' | 'pan'

    def press(self, x, y, ctrl):
        self._start = (x, y)
        self._mode  = 'pan' if ctrl else 'rotate'

    def drag(self, x, y):
        if self._start is None:
            return
        dx, dy     = x - self._start[0], y - self._start[1]
        self._start = (x, y)
        if self._mode == 'rotate':
            self.rot_y += dx * 0.40
            self.rot_x += dy * 0.40
        else:
            self.pan_x += dx * 0.025 / self.zoom
            self.pan_y -= dy * 0.025 / self.zoom

    def release(self):
        self._start = None

    def scroll(self, direction):
        self.zoom = max(0.08, min(12.0, self.zoom * (1.12 if direction > 0 else 0.89)))

    def apply_gl(self):
        glScalef(self.zoom, self.zoom, self.zoom)
        glTranslatef(self.pan_x, self.pan_y, 0.0)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)


# ══════════════════════════════════════════════════════════════════════════════
# OpenGL helpers
# ══════════════════════════════════════════════════════════════════════════════

def _gl_setup_lighting():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT0, GL_POSITION, (2.0,  3.0,  3.5,  0.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  (0.88, 0.88, 0.92, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (0.55, 0.55, 0.65, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT,  (0.0,  0.0,  0.0,  1.0))
    glLightfv(GL_LIGHT1, GL_POSITION, (-1.5, -1.5,  1.0, 0.0))
    glLightfv(GL_LIGHT1, GL_DIFFUSE,  (0.28, 0.28, 0.34, 1.0))
    glLightfv(GL_LIGHT1, GL_AMBIENT,  (0.10, 0.10, 0.13, 1.0))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glShadeModel(GL_FLAT)


def _draw_stent(verts, faces, normals, preview=False, scale_preview=1.0):
    if verts is None:
        return
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

    # Apply preview scale (radial approximation of diameter change)
    if abs(scale_preview - 1.0) > 0.001:
        glPushMatrix()
        glScalef(scale_preview, scale_preview, 1.0)  # radial only

    if preview:
        glColor4f(1.0, 0.88, 0.35, 0.90)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.5, 0.45, 0.2, 1.0))
    else:
        glColor4f(0.93, 0.93, 1.00, 1.00)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.45, 0.45, 0.55, 1.0))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 44.0)

    # Filled faces
    glBegin(GL_TRIANGLES)
    for i, face in enumerate(faces):
        n = normals[i]
        glNormal3f(float(n[0]), float(n[1]), float(n[2]))
        for vi in face:
            v = verts[vi]
            glVertex3f(float(v[0]), float(v[1]), float(v[2]))
    glEnd()

    # Subtle wireframe overlay
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glLineWidth(0.5)
    glColor4f(0.35, 0.40, 0.65, 0.22)
    glBegin(GL_TRIANGLES)
    for face in faces:
        for vi in face:
            v = verts[vi]
            glVertex3f(float(v[0]), float(v[1]), float(v[2]))
    glEnd()
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glDisable(GL_BLEND)
    glEnable(GL_LIGHTING)

    if abs(scale_preview - 1.0) > 0.001:
        glPopMatrix()

    glDisable(GL_DEPTH_TEST)


def _surface_to_texture(surface: pygame.Surface) -> int:
    data   = pygame.image.tostring(surface, "RGBA", True)
    w, h   = surface.get_size()
    tex    = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    return tex


def _draw_panel_texture(tex, view_w, win_w, win_h):
    """Blit panel texture over the right portion of the screen."""
    glViewport(0, 0, win_w, win_h)
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    glOrtho(0, win_w, 0, win_h, -1, 1)
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity()

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glBindTexture(GL_TEXTURE_2D, tex)
    glColor4f(1, 1, 1, 1)

    x1, x2 = float(view_w), float(win_w)
    y1, y2 = 0.0, float(win_h)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(x1, y1)
    glTexCoord2f(1, 0); glVertex2f(x2, y1)
    glTexCoord2f(1, 1); glVertex2f(x2, y2)
    glTexCoord2f(0, 1); glVertex2f(x1, y2)
    glEnd()

    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)


# ══════════════════════════════════════════════════════════════════════════════
# Slider widget
# ══════════════════════════════════════════════════════════════════════════════

class Slider:
    BAR_H   = 8
    KNOB_R  = 9
    LABEL_H = 17

    def __init__(self, key, label, lo, hi, step, fmt, rect: pygame.Rect):
        self.key      = key
        self.label    = label
        self.lo, self.hi, self.step = lo, hi, step
        self.fmt      = fmt
        self.rect     = rect        # full row rect in panel-local coords
        self.dragging = False
        self.editing  = False
        self.edit_buf = ""

    @property
    def bar_rect(self) -> pygame.Rect:
        r = self.rect
        y = r.y + self.LABEL_H + (r.h - self.LABEL_H - self.BAR_H) // 2
        return pygame.Rect(r.x, y, r.w, self.BAR_H)

    def val_to_x(self, val) -> float:
        br = self.bar_rect
        t  = (val - self.lo) / (self.hi - self.lo)
        return br.x + t * br.w

    def x_to_val(self, x) -> float:
        br  = self.bar_rect
        t   = max(0.0, min(1.0, (x - br.x) / br.w))
        raw = self.lo + t * (self.hi - self.lo)
        return max(self.lo, min(self.hi, round(raw / self.step) * self.step))

    def knob_hit(self, mx, my, val) -> bool:
        kx = self.val_to_x(val)
        ky = self.bar_rect.centery
        return (mx - kx) ** 2 + (my - ky) ** 2 <= (self.KNOB_R + 4) ** 2

    def bar_hit(self, mx, my) -> bool:
        br = self.bar_rect
        return br.inflate(0, 10).collidepoint(mx, my)


# ══════════════════════════════════════════════════════════════════════════════
# UI Panel (rendered to a pygame Surface, uploaded as GL texture each frame)
# ══════════════════════════════════════════════════════════════════════════════

class UIPanel:
    PAD    = 14
    ROW_H  = 50     # height of one slider row
    BTN_H  = 40
    SEP    = 10

    def __init__(self, w, h, settings, frame_cache):
        self.w = w
        self.h = h
        self.committed  = copy.deepcopy(settings)
        self.preview    = copy.deepcopy(settings)
        self.frame_cache = frame_cache
        self.cur_frame   = 0
        self.dirty       = False     # preview ≠ committed
        self.sim_running = False
        self.status_msg  = ""
        self.status_col  = C_TEXT_DIM
        self.hover       = None

        self._fonts_ok   = False
        self._font_lg = self._font_md = self._font_sm = None
        self.sliders: list[Slider] = []
        self._layout_done = False

    # ── lazy font & layout init (called after pygame.init) ────────────────────

    def _ensure_init(self):
        if self._layout_done:
            return
        try:
            self._font_lg = pygame.font.SysFont("Segoe UI",  17, bold=True)
            self._font_md = pygame.font.SysFont("Segoe UI",  14)
            self._font_sm = pygame.font.SysFont("Segoe UI",  13)
        except Exception:
            self._font_lg = pygame.font.SysFont(None, 19, bold=True)
            self._font_md = pygame.font.SysFont(None, 16)
            self._font_sm = pygame.font.SysFont(None, 14)
        self._build_layout()
        self._layout_done = True

    def _build_layout(self):
        PAD, ROW_H, BTN_H, SEP = self.PAD, self.ROW_H, self.BTN_H, self.SEP
        w = self.w - 2 * PAD
        y = PAD

        self._title_y = y;          y += 22
        self._banner_y = y;         y += 18   # preview banner

        y += SEP
        # Frame nav
        nb = 44
        self._prev_btn  = pygame.Rect(PAD,          y, nb, BTN_H)
        self._next_btn  = pygame.Rect(PAD + nb + 4, y, nb, BTN_H)
        self._frame_lbl = pygame.Rect(PAD + nb*2+10, y, w - nb*2 - 10, BTN_H)
        y += BTN_H + SEP

        # Separator
        self._sep1 = y;             y += SEP + 2

        # Params header
        self._ph_y = y;             y += 20 + 4

        # Sliders
        self.sliders = []
        for key, label, lo, hi, step, fmt in PARAM_SPECS:
            rect = pygame.Rect(PAD, y, w, ROW_H)
            self.sliders.append(Slider(key, label, lo, hi, step, fmt, rect))
            y += ROW_H + 6

        # Separator
        self._sep2 = y;             y += SEP + 2

        # Buttons
        half = (w - 6) // 2
        self._regen_btn = pygame.Rect(PAD,          y, half, BTN_H)
        self._save_btn  = pygame.Rect(PAD+half+6,   y, half, BTN_H)
        y += BTN_H + SEP

        # Status
        self._status_y = y

    # ── rendering ─────────────────────────────────────────────────────────────

    def render(self) -> pygame.Surface:
        self._ensure_init()
        surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)

        bg = (*C_DIRTY_BG, 235) if self.dirty else (*C_PANEL, 240)
        surf.fill(bg)
        pygame.draw.line(surf, C_PANEL_EDGE, (0, 0), (0, self.h), 2)

        # Title
        self._txt(surf, self._font_lg, "Stent FEA Viewer", C_ACCENT,
                  self.PAD, self._title_y)

        # Preview banner
        if self.dirty:
            self._txt(surf, self._font_sm, "  PREVIEW — uncommitted changes",
                      C_PREVIEW, self.PAD, self._banner_y)
        else:
            self._txt(surf, self._font_sm, "  Settings committed",
                      C_TEXT_DIM, self.PAD, self._banner_y)

        # Separator
        pygame.draw.line(surf, C_PANEL_EDGE,
                         (self.PAD, self._sep1), (self.w - self.PAD, self._sep1), 1)

        # Frame navigator
        n = max(1, self.frame_cache.count)
        self._btn(surf, self._prev_btn,  "<",    C_BTN_FRAME, C_BTN_FRAME_H, "prev")
        self._btn(surf, self._next_btn,  ">",    C_BTN_FRAME, C_BTN_FRAME_H, "next")
        label = f"Frame  {self.cur_frame + 1}  /  {n}"
        lt = self._font_md.render(label, True, C_TEXT)
        surf.blit(lt, (self._frame_lbl.x + 6,
                       self._frame_lbl.y + (self.BTN_H - lt.get_height()) // 2))

        # Params header
        self._txt(surf, self._font_md, "Simulation Parameters", C_TEXT,
                  self.PAD, self._ph_y)

        # Sliders
        for sl in self.sliders:
            self._draw_slider(surf, sl, self.preview.get(sl.key, sl.lo))

        # Separator
        pygame.draw.line(surf, C_PANEL_EDGE,
                         (self.PAD, self._sep2), (self.w - self.PAD, self._sep2), 1)

        # Buttons
        rlbl = "Regenerating..." if self.sim_running else "Regenerate"
        self._btn(surf, self._regen_btn, rlbl,
                  C_BTN_REGEN, C_BTN_REGEN_H, "regen", self.sim_running)
        self._btn(surf, self._save_btn,  "Save Settings",
                  C_BTN_SAVE,  C_BTN_SAVE_H,  "save")

        # Status
        if self.status_msg:
            self._txt(surf, self._font_sm, self.status_msg,
                      self.status_col, self.PAD, self._status_y)

        return surf

    def _txt(self, surf, font, text, color, x, y):
        t = font.render(text, True, color)
        surf.blit(t, (x, y))

    def _btn(self, surf, rect, label, col, col_h, name, disabled=False):
        c = tuple(max(0, v - 25) for v in col) if disabled \
            else (col_h if self.hover == name else col)
        pygame.draw.rect(surf, c, rect, border_radius=6)
        pygame.draw.rect(surf, C_PANEL_EDGE, rect, 1, border_radius=6)
        tc = C_TEXT_DIM if disabled else C_TEXT
        t  = self._font_sm.render(label, True, tc)
        surf.blit(t, (rect.x + (rect.w - t.get_width()) // 2,
                      rect.y + (rect.h - t.get_height()) // 2))

    def _draw_slider(self, surf, sl: Slider, val):
        r  = sl.rect
        br = sl.bar_rect

        # Label (left) + value (right)
        self._txt(surf, self._font_sm, sl.label, C_TEXT_DIM, r.x, r.y)
        if sl.editing:
            vs, vc = sl.edit_buf + "|", C_PREVIEW
        else:
            vs, vc = sl.fmt.format(val), C_ACCENT
        vt = self._font_sm.render(vs, True, vc)
        surf.blit(vt, (r.right - vt.get_width(), r.y))

        # Track
        pygame.draw.rect(surf, C_SLIDER_BG, br, border_radius=4)
        t     = (val - sl.lo) / (sl.hi - sl.lo)
        fw    = max(0, int(t * br.w))
        if fw:
            pygame.draw.rect(surf, C_SLIDER_FG,
                             pygame.Rect(br.x, br.y, fw, br.h), border_radius=4)

        # Knob
        kx = int(br.x + t * br.w)
        ky = br.centery
        pygame.draw.circle(surf, C_KNOB,   (kx, ky), sl.KNOB_R)
        pygame.draw.circle(surf, C_ACCENT, (kx, ky), sl.KNOB_R, 2)

    # ── event handling (mx, my in panel-local coords) ────────────────────────

    def handle(self, event, offset_x: int) -> bool:
        """Returns True if event was consumed."""
        self._ensure_init()

        def local(pos):
            return pos[0] - offset_x, pos[1]

        if event.type == MOUSEMOTION:
            mx, my = local(event.pos)
            self.hover = self._hit(mx, my)
            for sl in self.sliders:
                if sl.dragging:
                    self.preview[sl.key] = sl.x_to_val(mx)
                    self.dirty = True
            return False

        if event.type == MOUSEBUTTONDOWN:
            mx, my = local(event.pos)

            # Right-click on value text → enter edit mode
            if event.button == 3:
                vt_w = 60   # approximate value text width
                for sl in self.sliders:
                    if pygame.Rect(sl.rect.right - vt_w, sl.rect.y,
                                   vt_w, sl.Slider.LABEL_H if hasattr(sl, 'Slider') else Slider.LABEL_H
                                   ).collidepoint(mx, my):
                        sl.editing  = True
                        sl.edit_buf = sl.fmt.format(self.preview.get(sl.key, sl.lo))
                        return True
                # Simplified: right-click anywhere on label row
                for sl in self.sliders:
                    if sl.rect.collidepoint(mx, my) and not sl.bar_hit(mx, my):
                        sl.editing  = True
                        sl.edit_buf = sl.fmt.format(self.preview.get(sl.key, sl.lo))
                        return True

            if event.button == 1:
                # Sliders
                for sl in self.sliders:
                    if sl.knob_hit(mx, my, self.preview.get(sl.key, sl.lo)):
                        sl.dragging = True;  return True
                    if sl.bar_hit(mx, my):
                        sl.dragging = True
                        self.preview[sl.key] = sl.x_to_val(mx)
                        self.dirty = True;   return True
                # Frame nav
                if self._prev_btn.collidepoint(mx, my):
                    self.cur_frame = max(0, self.cur_frame - 1);  return True
                if self._next_btn.collidepoint(mx, my):
                    self.cur_frame = min(self.frame_cache.count-1,
                                        self.cur_frame + 1);      return True
                if self._regen_btn.collidepoint(mx, my) and not self.sim_running:
                    self._start_regen();                           return True
                if self._save_btn.collidepoint(mx, my):
                    save_settings(self.committed)
                    self.status_msg = "Settings saved."
                    self.status_col = C_OK;                        return True
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
                    elif event.key == K_ESCAPE:
                        sl.editing = False
                    elif event.key == K_BACKSPACE:
                        sl.edit_buf = sl.edit_buf[:-1]
                    elif event.unicode in "0123456789.-":
                        sl.edit_buf += event.unicode
                    return True
        return False

    def _hit(self, mx, my) -> str:
        for name, rect in [("prev",  self._prev_btn),
                            ("next",  self._next_btn),
                            ("regen", self._regen_btn),
                            ("save",  self._save_btn)]:
            if rect.collidepoint(mx, my):
                return name
        return None

    # ── simulation runner ────────────────────────────────────────────────────

    def _start_regen(self):
        self.committed    = copy.deepcopy(self.preview)
        self.dirty        = False
        self.sim_running  = True
        self.status_msg   = "Simulation running..."
        self.status_col   = C_ACCENT
        threading.Thread(target=self._regen_worker, daemon=True).start()

    def _regen_worker(self):
        s    = self.committed
        base = Path(__file__).parent
        cmd  = [
            sys.executable, str(base / "simulate.py"),
            "--input",             str(base / s["input"]),
            "--crimp-diameter",    str(s["crimp_diameter"]),
            "--deployed-diameter", str(s["deployed_diameter"]),
            "--output-dir",        str(base / s["output_dir"]),
            "--n-crimp-steps",     str(int(s["n_crimp_steps"])),
            "--n-deploy-steps",    str(int(s["n_deploy_steps"])),
            "--transition-length", str(s["transition_length"]),
            "--snap-speed",        str(s["snap_speed"]),
            "--crown-dwell",       str(s["crown_dwell"]),
            "--expansion-exponent",str(s["expansion_exponent"]),
            "--tine-flare",        str(s["tine_flare"]),
        ]
        try:
            r = subprocess.run(cmd, cwd=str(base),
                               capture_output=True, text=True, timeout=600)
            if r.returncode == 0:
                self.frame_cache.refresh()
                self.cur_frame   = 0
                self.status_msg  = f"Done  —  {self.frame_cache.count} frames."
                self.status_col  = C_OK
            else:
                self.status_msg = "Simulation failed — see console."
                self.status_col = C_ERR
                print(r.stderr[-2000:])
        except Exception as e:
            self.status_msg = f"Error: {e}"
            self.status_col = C_ERR
        finally:
            self.sim_running = False

    # ── preview scale helper ─────────────────────────────────────────────────

    @property
    def preview_scale(self) -> float:
        """Radial scale to approximate the diameter change in preview mode."""
        if not self.dirty:
            return 1.0
        c_d = self.committed.get("deployed_diameter", 28.0)
        p_d = self.preview.get("deployed_diameter", 28.0)
        if c_d <= 0:
            return 1.0
        return p_d / c_d


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

def main(frames_dir: str = "./frames"):
    pygame.init()
    pygame.display.set_caption("Stent FEA Viewer")
    screen = pygame.display.set_mode((WIN_W, WIN_H), DOUBLEBUF | OPENGL)

    # Global GL state
    glClearColor(*[c / 255.0 for c in C_BG], 1.0)
    glEnable(GL_NORMALIZE)
    _gl_setup_lighting()

    # Core objects
    settings    = load_settings()
    frame_cache = FrameCache(frames_dir)
    starfield   = StarField()
    camera      = Camera()
    panel       = UIPanel(PANEL_W, WIN_H, settings, frame_cache)
    panel._ensure_init()

    panel_tex   = None
    clock       = pygame.time.Clock()
    t_prev      = time.time()

    while True:
        now = time.time()
        dt  = min(now - t_prev, 0.05)
        t_prev = now

        ctrl = bool(pygame.key.get_mods() & KMOD_CTRL)

        # ── event loop ────────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == QUIT:
                _cleanup(panel_tex); pygame.quit(); return

            if ev.type == KEYDOWN:
                if ev.key == K_ESCAPE:
                    _cleanup(panel_tex); pygame.quit(); return
                if ev.key == K_LEFT:
                    panel.cur_frame = max(0, panel.cur_frame - 1)
                if ev.key == K_RIGHT:
                    panel.cur_frame = min(frame_cache.count - 1,
                                         panel.cur_frame + 1)
                panel.handle(ev, VIEW_W)

            elif ev.type == MOUSEBUTTONDOWN:
                mx = ev.pos[0]
                if mx < VIEW_W:
                    camera.press(ev.pos[0], ev.pos[1], ctrl)
                else:
                    panel.handle(ev, VIEW_W)

            elif ev.type == MOUSEBUTTONUP:
                camera.release()
                panel.handle(ev, VIEW_W)

            elif ev.type == MOUSEMOTION:
                if ev.pos[0] < VIEW_W:
                    camera.drag(ev.pos[0], ev.pos[1])
                else:
                    panel.handle(ev, VIEW_W)

            elif ev.type == MOUSEWHEEL:
                if pygame.mouse.get_pos()[0] < VIEW_W:
                    camera.scroll(ev.y)

        # ── update ────────────────────────────────────────────────────────────
        starfield.update(dt)

        # ── render ────────────────────────────────────────────────────────────
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 1 ─ Star field (orthographic, full 3-D viewport) ────────────────────
        glViewport(0, 0, VIEW_W, WIN_H)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW);  glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        starfield.draw()

        # 2 ─ 3-D stent ───────────────────────────────────────────────────────
        verts, faces, normals, ctr, scl = frame_cache.get(panel.cur_frame)

        glViewport(0, 0, VIEW_W, WIN_H)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(42.0, VIEW_W / WIN_H, 0.1, 800.0)
        glMatrixMode(GL_MODELVIEW);  glLoadIdentity()
        glTranslatef(0.0, 0.0, -55.0)
        camera.apply_gl()

        if verts is not None and scl is not None:
            glScalef(scl, scl, scl)
            glTranslatef(-float(ctr[0]), -float(ctr[1]), -float(ctr[2]))
            _draw_stent(verts, faces, normals,
                        preview=panel.dirty,
                        scale_preview=panel.preview_scale)
        else:
            # No frames — draw a placeholder message via panel
            pass

        # 3 ─ UI panel (pygame surface → GL texture) ──────────────────────────
        if panel_tex is not None:
            try:
                glDeleteTextures(1, [panel_tex])
            except Exception:
                pass
        panel_surf = panel.render()
        panel_tex  = _surface_to_texture(panel_surf)
        _draw_panel_texture(panel_tex, VIEW_W, WIN_W, WIN_H)

        pygame.display.flip()
        clock.tick(TARGET_FPS)


def _cleanup(tex):
    if tex is not None:
        try:
            glDeleteTextures(1, [tex])
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fd = sys.argv[1] if len(sys.argv) > 1 else "./frames"
    main(frames_dir=fd)
