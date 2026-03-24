"""
viewer.py — Interactive 3D Stent FEA Viewer  (v3)

All parameters update the stent mesh in real time using the same cylindrical
deformation maths as deform.py.  No simulation re-run needed for preview.

Controls
--------
  Left-drag          : Rotate        Scroll : Zoom
  Ctrl + Left-drag   : Pan           ← / →  : Prev / next frame
  Right-click slider : Type value + Enter
  Click frame number : Type frame + Enter
  ▶ / ⏸             : Play / pause at 30 fps
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
    print("[viewer] pip install PyOpenGL PyOpenGL_accelerate"); sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# Constants & colours
# ══════════════════════════════════════════════════════════════════════════════

WIN_W, WIN_H  = 1440, 900
PANEL_W       = 430
VIEW_W        = WIN_W - PANEL_W
FPS           = 60
PLAY_FPS      = 30
DEFORM_HZ     = 30          # max deformation updates per second

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

PARAM_SPECS = [
    # key, label, lo, hi, step, fmt
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

C_BG         = (12,  12,  30);   C_PANEL      = (20,  20,  50)
C_PANEL_EDGE = (60,  65, 160);   C_TEXT       = (215, 220, 255)
C_TEXT_DIM   = (115, 120, 170);  C_ACCENT     = (100, 140, 255)
C_SLIDER_BG  = (38,  40,  88);   C_SLIDER_FG  = (85, 120, 245)
C_KNOB       = (195, 210, 255);  C_BTN_FRAME  = (50,  50, 110)
C_BTN_FRAMEH = (75,  75, 160);   C_BTN_REGEN  = (38, 115,  55)
C_BTN_RGENH  = (52, 155,  72);   C_BTN_SAVE   = (45,  70, 150)
C_BTN_SAVEH  = (60,  92, 195);   C_PREVIEW    = (255, 200,  45)
C_DIRTY_BG   = (32,  28,   8);   C_OK         = (80, 210, 100)
C_ERR        = (255,  70,  70);  C_INPUT_BG   = (30,  32,  72)


# ══════════════════════════════════════════════════════════════════════════════
# Settings
# ══════════════════════════════════════════════════════════════════════════════

def load_settings():
    s = copy.deepcopy(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            d = json.loads(SETTINGS_FILE.read_text())
            for k in s:
                if k in d: s[k] = d[k]
        except Exception: pass
    return s

def save_settings(s):
    try: SETTINGS_FILE.write_text(json.dumps(s, indent=2))
    except Exception as e: print(f"[viewer] save: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# FramePlayer — anchored deformation + VBO management
#
# Loads frame_000 (crimped) and frame_099 (deployed) as cylindrical-coord
# anchors, then reconstructs any parameter/frame combination via the same
# maths as deform.py — all in numpy, uploaded via glBufferSubData.
# ══════════════════════════════════════════════════════════════════════════════

def _smoothstep(x):
    t = np.clip(x, 0., 1.); return t*t*(3.-2.*t)

def _snap_curve(t, speed):
    return 1. - (1. - t) ** max(speed, 0.01)


class FramePlayer:
    """Manages frame paths, anchor geometry, and the single GPU VBO."""

    def __init__(self, frames_dir: str):
        self.dir    = Path(frames_dir)
        self.paths  = []
        self.vbo    = None
        self.n_draw = 0
        self.centre = np.zeros(3, np.float32)
        self.scale  = 1.0

        # Anchor cylindrical coords (computed once from frame_000)
        self._theta      = None   # (N,)  frozen azimuth
        self._z          = None   # (N,)  frozen z
        self._r_offset   = None   # (N,)  cross-section offset
        self._r_center   = 0.     # median radius of crimped state
        self._cx = self._cy = 0.
        self._faces      = None   # (M, 3)
        self._cell_h     = 1.     # estimated crown cell height

        self._gl_ready   = False
        self._last_key   = None
        self._last_deform_t = 0.

        self._scan()

    # ── file management ───────────────────────────────────────────────────────

    def _scan(self):
        self.paths = sorted(self.dir.glob("frame_*.stl"))

    def refresh(self):
        self._scan()
        self._gl_ready = False
        self._last_key = None

    @property
    def count(self): return len(self.paths)

    # ── GL init (call after OpenGL context is ready) ──────────────────────────

    def ensure_ready(self):
        if self._gl_ready or not self.paths:
            return
        try:
            self._load_anchors()
            self._gl_ready = True
        except Exception as e:
            print(f"[viewer] anchor load: {e}")

    def _load_anchors(self):
        m0 = trimesh.load(str(self.paths[0]),   process=False, force='mesh')
        m1 = trimesh.load(str(self.paths[-1]),  process=False, force='mesh')

        if len(m0.vertices) != len(m1.vertices):
            print("[viewer] warn: anchor vertex counts differ — using frame 0 only")
            m1 = m0

        v0  = m0.vertices.astype(np.float64)
        v1  = m1.vertices.astype(np.float64)
        f   = m0.faces.astype(np.int32)

        # Axis centre from crimped frame
        cx  = float(np.median(v0[:, 0]))
        cy  = float(np.median(v0[:, 1]))
        self._cx, self._cy = cx, cy

        dx  = v0[:, 0] - cx;  dy = v0[:, 1] - cy
        r0  = np.sqrt(dx**2 + dy**2)

        self._theta    = np.arctan2(dy, dx).astype(np.float32)
        self._z        = v0[:, 2].astype(np.float32)
        self._r_center = float(np.median(r0))
        self._r_offset = (r0 - self._r_center).astype(np.float32)
        self._faces    = f

        # Estimate crown cell height from z-value clustering
        z_vals  = np.sort(v0[:, 2])
        z_span  = float(z_vals[-1] - z_vals[0])
        diffs   = np.diff(z_vals)
        thresh  = np.percentile(diffs, 88)
        gaps    = diffs[diffs > thresh]
        self._cell_h = float(np.mean(gaps)) * 2.0 if len(gaps) > 1 else z_span / 5.

        # Fit scale / centre from deployed frame
        v1f = v1
        self.centre = ((v1f.min(0) + v1f.max(0)) * .5).astype(np.float32)
        ext = float((v1f.max(0) - v1f.min(0)).max())
        self.scale  = 20. / max(ext, 1e-6)

        # Pre-allocate VBO
        n = len(f) * 3
        if self.vbo is None:
            self.vbo = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, n * 24, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.n_draw = n

    # ── deformation update ────────────────────────────────────────────────────

    def update(self, frame_idx: int, preview: dict, committed: dict, now: float):
        """Recompute VBO if params/frame changed (throttled to DEFORM_HZ)."""
        if not self._gl_ready:
            return

        key = (
            frame_idx,
            round(preview.get("deployed_diameter",  28.), 2),
            round(preview.get("crimp_diameter",       6.), 2),
            round(preview.get("transition_length",   .45), 3),
            round(preview.get("snap_speed",          3.0), 2),
            round(preview.get("crown_dwell",          .6), 3),
            round(preview.get("expansion_exponent",   .6), 3),
        )
        if key == self._last_key:
            return
        if now - self._last_deform_t < 1. / DEFORM_HZ:
            return

        self._last_key      = key
        self._last_deform_t = now
        self._do_deform(frame_idx, preview, committed)

    def _do_deform(self, frame_idx: int, preview: dict, committed: dict):
        """Port of deform.py's cylindrical deformation, vectorised in numpy."""
        n_frames = max(self.count - 1, 1)
        t        = frame_idx / n_frames          # 0 = crimped, 1 = deployed

        # ── Parameters ───────────────────────────────────────────────────────
        crimp_r   = float(preview.get("crimp_diameter",      6.)) / 2.
        deploy_r  = float(preview.get("deployed_diameter",  28.)) / 2.
        tl        = max(0.01, float(preview.get("transition_length", .45)))
        snap      = max(0.01, float(preview.get("snap_speed",        3.0)))
        dwell     = max(0.,   float(preview.get("crown_dwell",        .6)))
        exp_e     = max(0.01, float(preview.get("expansion_exponent", .6)))

        z    = self._z.astype(np.float64)
        z_min, z_max = float(z.min()), float(z.max())
        z_span = max(z_max - z_min, 1e-6)

        # ── Crown dwell: per-vertex z shift (same formula as deform.py) ──────
        cell_h        = max(self._cell_h, 1e-6)
        crown_arm_len = cell_h / 2.
        z_in_cell     = (z - z_min) % cell_h
        z_cell_frac   = z_in_cell / cell_h
        crown_prox    = 2. * np.abs(z_cell_frac - 0.5)
        dwell_shift   = crown_arm_len * dwell * crown_prox
        z_eff         = z - dwell_shift

        # ── Global expansion factor ───────────────────────────────────────────
        global_exp = t ** exp_e
        r_max      = crimp_r + (deploy_r - crimp_r) * global_exp

        # ── Per-vertex released fraction (smoothstep over transition zone) ────
        tube_tip_z = z_max - t * z_span
        trans_len  = tl * z_span
        released   = _smoothstep((z_eff - tube_tip_z) / max(trans_len, 1e-6))
        snap_v     = _snap_curve(released, snap)

        # ── New centerline radius ─────────────────────────────────────────────
        r_cl  = crimp_r + (r_max - crimp_r) * snap_v
        r_new = (r_cl + self._r_offset).astype(np.float64)

        # ── Back to cartesian ─────────────────────────────────────────────────
        theta = self._theta.astype(np.float64)
        vx    = self._cx + r_new * np.cos(theta)
        vy    = self._cy + r_new * np.sin(theta)
        vz    = z
        verts = np.stack([vx, vy, vz], axis=1).astype(np.float32)

        # ── Face normals (vectorised cross product) ───────────────────────────
        f  = self._faces
        e1 = verts[f[:, 1]] - verts[f[:, 0]]
        e2 = verts[f[:, 2]] - verts[f[:, 0]]
        n  = np.cross(e1, e2)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n  = (n / np.maximum(ln, 1e-8)).astype(np.float32)

        # ── Pack [nx ny nz vx vy vz] × 3 per face ────────────────────────────
        v_exp = verts[f].reshape(-1, 3)              # (M*3, 3)
        n_exp = np.repeat(n, 3, axis=0)              # (M*3, 3)
        data  = np.ascontiguousarray(
            np.hstack([n_exp, v_exp]), dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    # ── draw ──────────────────────────────────────────────────────────────────

    def draw(self, preview: bool):
        if not self.vbo or not self.n_draw:
            return
        if preview:
            glColor4f(1.0, 0.87, 0.30, 0.92)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (.5, .44, .15, 1.))
        else:
            glColor4f(0.93, 0.94, 1.00, 1.00)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (.45, .45, .55, 1.))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 46.)

        stride = 24
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_VERTEX_ARRAY)
        glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(0))
        glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(12))

        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
        glDrawArrays(GL_TRIANGLES, 0, self.n_draw)

        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(0.6)
        glColor4f(.3, .35, .60, .16)
        glDrawArrays(GL_TRIANGLES, 0, self.n_draw)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDisable(GL_BLEND)

        glDisableClientState(GL_VERTEX_ARRAY); glDisableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisable(GL_DEPTH_TEST)

    def free(self):
        if self.vbo:
            try: glDeleteBuffers(1, [self.vbo])
            except Exception: pass
            self.vbo = None


# ══════════════════════════════════════════════════════════════════════════════
# Star field
# ══════════════════════════════════════════════════════════════════════════════

_STAR_COLS = [(1.,1.,1.),(.78,.82,1.),(1.,.93,.76),(.70,.78,1.),(1.,1.,.84),(.85,.7,1.)]

class StarField:
    N = 320
    def __init__(self):
        self.t    = 0.
        xs        = np.random.uniform(-1.05, 1.05, self.N).astype(np.float32)
        ys        = np.random.uniform(-1.05, 1.05, self.N).astype(np.float32)
        self._pos = np.stack([xs, ys], 1)
        self._col = np.array([random.choice(_STAR_COLS) for _ in range(self.N)], np.float32)
        self._ph  = np.random.uniform(0, 6.28, self.N).astype(np.float32)
        self._fr  = np.random.uniform(0.4, 3.5, self.N).astype(np.float32)
        tiers     = np.array([0]*28 + [1]*90 + [2]*(self.N-118))
        np.random.shuffle(tiers); self._tier = tiers
        self._comets = [_Comet() for _ in range(4)]

    def update(self, dt):
        self.t += dt
        for c in self._comets: c.update(dt)

    def draw(self):
        glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        alpha = (0.45 + 0.55*np.sin(self._ph + self.t*self._fr)**2).astype(np.float32)
        rgba  = np.ascontiguousarray(np.hstack([self._col, alpha[:,None]]), np.float32)
        glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_COLOR_ARRAY)
        for sz, tier in [(2.8,0),(1.7,1),(1.0,2)]:
            m = self._tier==tier
            if not m.any(): continue
            p = np.ascontiguousarray(self._pos[m]); c = np.ascontiguousarray(rgba[m])
            glPointSize(sz)
            glVertexPointer(2,GL_FLOAT,0,p); glColorPointer(4,GL_FLOAT,0,c)
            glDrawArrays(GL_POINTS,0,len(p))
        glDisableClientState(GL_COLOR_ARRAY); glDisableClientState(GL_VERTEX_ARRAY)
        glEnable(GL_POINT_SMOOTH)
        for c in self._comets:
            if not c.active: continue
            fade = max(0., 1.-c.life/c.max_life)
            r,g,b = c.col
            for i in range(c.TRAIL):
                fr = i/c.TRAIL; a=(1-fr)*fade*.9; sz=(1-fr)*3.6
                glColor4f(r,g,b,a); glPointSize(max(.4,sz))
                glBegin(GL_POINTS); glVertex2f(c.x-c.vx*fr*.09,c.y-c.vy*fr*.09); glEnd()
        glDisable(GL_POINT_SMOOTH)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDisable(GL_BLEND)


class _Comet:
    TRAIL=26; _C=[(1.,.4,.1),(.35,.55,1.),(.8,.28,1.),(.25,1.,.65),(1.,.9,.15),(1.,1.,1.),(1.,1.,1.)]
    def __init__(self):
        self.active=False; self.timer=0.; self.wait=random.uniform(2.,9.)
        self.x=self.y=self.vx=self.vy=self.life=self.max_life=0.; self.col=(1.,1.,1.)
    def spawn(self):
        e=random.randint(0,3)
        if e==0: self.x,self.y=random.uniform(-1,1),1.15
        elif e==1: self.x,self.y=random.uniform(-1,1),-1.15
        elif e==2: self.x,self.y=-1.15,random.uniform(-1,1)
        else: self.x,self.y=1.15,random.uniform(-1,1)
        a=math.atan2(-self.y,-self.x)+random.uniform(-.45,.45); s=random.uniform(.35,1.1)
        self.vx,self.vy=math.cos(a)*s,math.sin(a)*s
        self.life,self.max_life=0.,random.uniform(.7,1.8)
        self.col=random.choice(self._C); self.active=True
    def update(self,dt):
        self.timer+=dt
        if not self.active:
            if self.timer>=self.wait: self.timer=0.;self.wait=random.uniform(3.,12.);self.spawn()
            return
        self.life+=dt; self.x+=self.vx*dt; self.y+=self.vy*dt
        if abs(self.x)>1.6 or abs(self.y)>1.6 or self.life>self.max_life:
            self.active=False; self.timer=0.; self.wait=random.uniform(2.,9.)


# ══════════════════════════════════════════════════════════════════════════════
# Camera
# ══════════════════════════════════════════════════════════════════════════════

class Camera:
    def __init__(self):
        self.rx=18.; self.ry=-25.; self.zoom=1.; self.px=self.py=0.
        self._p=self._mode=None
    def press(self,x,y,ctrl): self._p=(x,y); self._mode='pan' if ctrl else 'rot'
    def drag(self,x,y):
        if not self._p: return
        dx,dy=x-self._p[0],y-self._p[1]; self._p=(x,y)
        if self._mode=='rot': self.ry+=dx*.40; self.rx+=dy*.40
        else: self.px+=dx*.025/self.zoom; self.py-=dy*.025/self.zoom
    def release(self): self._p=None
    def scroll(self,d): self.zoom=max(.08,min(12.,self.zoom*(1.12 if d>0 else .89)))
    def apply(self):
        glScalef(self.zoom,self.zoom,self.zoom); glTranslatef(self.px,self.py,0.)
        glRotatef(self.rx,1,0,0); glRotatef(self.ry,0,1,0)


# ══════════════════════════════════════════════════════════════════════════════
# OpenGL helpers
# ══════════════════════════════════════════════════════════════════════════════

def setup_lighting():
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT0,GL_POSITION,(2.,3.,3.5,0.)); glLightfv(GL_LIGHT0,GL_DIFFUSE,(.88,.88,.92,1.))
    glLightfv(GL_LIGHT0,GL_SPECULAR,(.55,.55,.65,1.)); glLightfv(GL_LIGHT0,GL_AMBIENT,(0.,0.,0.,1.))
    glLightfv(GL_LIGHT1,GL_POSITION,(-1.5,-1.5,1.,0.)); glLightfv(GL_LIGHT1,GL_DIFFUSE,(.28,.28,.34,1.))
    glLightfv(GL_LIGHT1,GL_AMBIENT,(.10,.10,.13,1.))
    glEnable(GL_COLOR_MATERIAL); glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE)
    glShadeModel(GL_FLAT)

def surf_to_tex(surf):
    data=pygame.image.tostring(surf,"RGBA",True); w,h=surf.get_size()
    t=int(glGenTextures(1)); glBindTexture(GL_TEXTURE_2D,t)
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,w,h,0,GL_RGBA,GL_UNSIGNED_BYTE,data)
    return t

def draw_panel_tex(tex,vw,ww,wh):
    glViewport(0,0,ww,wh)
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0,ww,0,wh,-1,1)
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity()
    glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D); glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
    glBindTexture(GL_TEXTURE_2D,tex); glColor4f(1,1,1,1)
    x1,x2,y1,y2=float(vw),float(ww),0.,float(wh)
    glBegin(GL_QUADS)
    glTexCoord2f(0,0); glVertex2f(x1,y1); glTexCoord2f(1,0); glVertex2f(x2,y1)
    glTexCoord2f(1,1); glVertex2f(x2,y2); glTexCoord2f(0,1); glVertex2f(x1,y2)
    glEnd()
    glDisable(GL_TEXTURE_2D); glDisable(GL_BLEND)


# ══════════════════════════════════════════════════════════════════════════════
# Slider widget
# ══════════════════════════════════════════════════════════════════════════════

class Slider:
    BAR_H=8; KNOB_R=9; LBL_H=17
    def __init__(self,key,label,lo,hi,step,fmt,rect):
        self.key,self.label=key,label; self.lo,self.hi,self.step=lo,hi,step
        self.fmt,self.rect=fmt,rect; self.dragging=False; self.editing=False; self.edit_buf=""
    @property
    def bar(self):
        r=self.rect; y=r.y+self.LBL_H+(r.h-self.LBL_H-self.BAR_H)//2
        return pygame.Rect(r.x,y,r.w,self.BAR_H)
    def val_x(self,v):
        br=self.bar; return br.x+(v-self.lo)/(self.hi-self.lo)*br.w
    def x_val(self,x):
        br=self.bar; t=max(0.,min(1.,(x-br.x)/br.w))
        raw=self.lo+t*(self.hi-self.lo)
        return max(self.lo,min(self.hi,round(raw/self.step)*self.step))
    def knob_hit(self,mx,my,v):
        kx,ky=self.val_x(v),self.bar.centery; return (mx-kx)**2+(my-ky)**2<=(self.KNOB_R+5)**2
    def bar_hit(self,mx,my): return self.bar.inflate(0,12).collidepoint(mx,my)


# ══════════════════════════════════════════════════════════════════════════════
# UI Panel
# ══════════════════════════════════════════════════════════════════════════════

class UIPanel:
    PAD=14; ROW_H=50; BTN_H=40

    def __init__(self, w, h, settings, player):
        self.w,self.h=w,h
        self.committed   = copy.deepcopy(settings)
        self.preview     = copy.deepcopy(settings)
        self.player      = player
        self.cur_frame   = 0
        self.dirty       = False
        self.sim_running = False
        self.status_msg  = ""
        self.status_col  = C_TEXT_DIM
        self.hover       = None
        self.needs_redraw= True
        self.playing     = False
        self._play_acc   = 0.        # accumulated time for play
        self._frame_editing = False  # typing in frame number box
        self._frame_buf  = ""
        self.sliders     = []
        self._fl = self._fm = self._fs = None
        self._ready      = False

    def init(self):
        try:
            self._fl=pygame.font.SysFont("Segoe UI",17,bold=True)
            self._fm=pygame.font.SysFont("Segoe UI",14)
            self._fs=pygame.font.SysFont("Segoe UI",13)
        except Exception:
            self._fl=pygame.font.SysFont(None,19,bold=True)
            self._fm=pygame.font.SysFont(None,16)
            self._fs=pygame.font.SysFont(None,14)
        PAD,ROW_H,BTN_H=self.PAD,self.ROW_H,self.BTN_H
        w=self.w-2*PAD; y=PAD
        self._title_y=y; y+=22
        self._banner_y=y; y+=18+8

        # Frame nav row: [<] [>] [▶/⏸] [frame_input]
        nb=44; pb=44
        self._prev_btn  = pygame.Rect(PAD,        y,nb,BTN_H)
        self._next_btn  = pygame.Rect(PAD+nb+4,   y,nb,BTN_H)
        self._play_btn  = pygame.Rect(PAD+nb*2+8, y,pb,BTN_H)
        input_x         = PAD+nb*2+pb+14
        self._frame_input=pygame.Rect(input_x, y, w-(nb*2+pb+14), BTN_H)
        y+=BTN_H+10

        self._sep1=y; y+=12
        self._ph_y=y;  y+=22
        self.sliders=[]
        for key,label,lo,hi,step,fmt in PARAM_SPECS:
            self.sliders.append(Slider(key,label,lo,hi,step,fmt,pygame.Rect(PAD,y,w,ROW_H)))
            y+=ROW_H+6
        self._sep2=y; y+=12
        half=(w-6)//2
        self._regen_btn=pygame.Rect(PAD,       y,half,BTN_H)
        self._save_btn =pygame.Rect(PAD+half+6,y,half,BTN_H)
        y+=BTN_H+8; self._status_y=y
        self._ready=True

    # ── tick (advance playback) ───────────────────────────────────────────────

    def tick(self, dt: float):
        if not self.playing or self.player.count < 2:
            return
        self._play_acc += dt
        interval = 1. / PLAY_FPS
        if self._play_acc >= interval:
            self._play_acc -= interval
            nxt = self.cur_frame + 1
            if nxt >= self.player.count:
                nxt = 0
            self.cur_frame    = nxt
            self.needs_redraw = True

    # ── render ────────────────────────────────────────────────────────────────

    def render(self):
        s=pygame.Surface((self.w,self.h),pygame.SRCALPHA)
        bg=(*C_DIRTY_BG,240) if self.dirty else (*C_PANEL,245)
        s.fill(bg)
        pygame.draw.line(s,C_PANEL_EDGE,(0,0),(0,self.h),2)
        T=self._t

        T(s,self._fl,"Stent FEA Viewer",C_ACCENT,self.PAD,self._title_y)
        if self.dirty:
            T(s,self._fs,"  PREVIEW — approximate real-time deformation",C_PREVIEW,self.PAD,self._banner_y)
        else:
            T(s,self._fs,"  Settings committed",C_TEXT_DIM,self.PAD,self._banner_y)

        pygame.draw.line(s,C_PANEL_EDGE,(self.PAD,self._sep1),(self.w-self.PAD,self._sep1),1)

        # Frame nav
        n=max(1,self.player.count)
        self._btn(s,self._prev_btn,"<",C_BTN_FRAME,C_BTN_FRAMEH,"prev")
        self._btn(s,self._next_btn,">",C_BTN_FRAME,C_BTN_FRAMEH,"next")
        play_lbl = "||" if self.playing else ">"
        play_col = C_BTN_RGENH if self.playing else C_BTN_REGEN
        self._btn(s,self._play_btn,play_lbl,play_col,C_BTN_RGENH,"play")

        # Frame number input box
        r=self._frame_input
        pygame.draw.rect(s,C_INPUT_BG,r,border_radius=5)
        pygame.draw.rect(s,C_PANEL_EDGE,r,1,border_radius=5)
        if self._frame_editing:
            ft=self._fs.render(self._frame_buf+"|",True,C_PREVIEW)
        else:
            ft=self._fm.render(f"{self.cur_frame+1} / {n}",True,C_TEXT)
        s.blit(ft,(r.x+(r.w-ft.get_width())//2, r.y+(r.h-ft.get_height())//2))

        T(s,self._fm,"Simulation Parameters",C_TEXT,self.PAD,self._ph_y)
        for sl in self.sliders:
            self._draw_slider(s,sl,self.preview.get(sl.key,sl.lo))

        pygame.draw.line(s,C_PANEL_EDGE,(self.PAD,self._sep2),(self.w-self.PAD,self._sep2),1)

        rl="Regenerating..." if self.sim_running else "Regenerate"
        self._btn(s,self._regen_btn,rl,C_BTN_REGEN,C_BTN_RGENH,"regen",self.sim_running)
        self._btn(s,self._save_btn,"Save Settings",C_BTN_SAVE,C_BTN_SAVEH,"save")
        if self.status_msg: T(s,self._fs,self.status_msg,self.status_col,self.PAD,self._status_y)

        self.needs_redraw=False
        return s

    def _t(self,s,f,txt,c,x,y): s.blit(f.render(txt,True,c),(x,y))

    def _btn(self,s,r,lbl,col,colh,name,disabled=False):
        c=tuple(max(0,v-25) for v in col) if disabled else (colh if self.hover==name else col)
        pygame.draw.rect(s,c,r,border_radius=6)
        pygame.draw.rect(s,C_PANEL_EDGE,r,1,border_radius=6)
        t=self._fs.render(lbl,True,C_TEXT_DIM if disabled else C_TEXT)
        s.blit(t,(r.x+(r.w-t.get_width())//2,r.y+(r.h-t.get_height())//2))

    def _draw_slider(self,s,sl,val):
        r,br=sl.rect,sl.bar
        self._t(s,self._fs,sl.label,C_TEXT_DIM,r.x,r.y)
        if sl.editing: vs,vc=sl.edit_buf+"|",C_PREVIEW
        else: vs,vc=sl.fmt.format(val),C_ACCENT
        vt=self._fs.render(vs,True,vc); s.blit(vt,(r.right-vt.get_width(),r.y))
        pygame.draw.rect(s,C_SLIDER_BG,br,border_radius=4)
        t=(val-sl.lo)/(sl.hi-sl.lo); fw=max(0,int(t*br.w))
        if fw: pygame.draw.rect(s,C_SLIDER_FG,pygame.Rect(br.x,br.y,fw,br.h),border_radius=4)
        kx,ky=int(sl.val_x(val)),br.centery
        pygame.draw.circle(s,C_KNOB,(kx,ky),sl.KNOB_R)
        pygame.draw.circle(s,C_ACCENT,(kx,ky),sl.KNOB_R,2)

    # ── events ────────────────────────────────────────────────────────────────

    def handle(self, ev, ox: int):
        if not self._ready: return False
        lx = lambda p: (p[0]-ox, p[1])

        if ev.type==MOUSEMOTION:
            mx,my=lx(ev.pos)
            old=self.hover; self.hover=self._hit(mx,my)
            if old!=self.hover: self.needs_redraw=True
            for sl in self.sliders:
                if sl.dragging:
                    nv=sl.x_val(mx)
                    if nv!=self.preview.get(sl.key):
                        self.preview[sl.key]=nv; self.dirty=True; self.needs_redraw=True
            return False

        if ev.type==MOUSEBUTTONDOWN:
            mx,my=lx(ev.pos)
            if ev.button==3:
                for sl in self.sliders:
                    if sl.rect.collidepoint(mx,my):
                        sl.editing=True; sl.edit_buf=sl.fmt.format(self.preview.get(sl.key,sl.lo))
                        self.needs_redraw=True; return True
            if ev.button==1:
                # Frame input box
                if self._frame_input.collidepoint(mx,my):
                    self._frame_editing=True; self._frame_buf=str(self.cur_frame+1)
                    self.needs_redraw=True; return True
                for sl in self.sliders:
                    v=self.preview.get(sl.key,sl.lo)
                    if sl.knob_hit(mx,my,v) or sl.bar_hit(mx,my):
                        sl.dragging=True; nv=sl.x_val(mx)
                        if nv!=v: self.preview[sl.key]=nv; self.dirty=True; self.needs_redraw=True
                        return True
                if self._prev_btn.collidepoint(mx,my):
                    self.cur_frame=max(0,self.cur_frame-1); self.needs_redraw=True; return True
                if self._next_btn.collidepoint(mx,my):
                    self.cur_frame=min(self.player.count-1,self.cur_frame+1); self.needs_redraw=True; return True
                if self._play_btn.collidepoint(mx,my):
                    self.playing=not self.playing; self._play_acc=0.; self.needs_redraw=True; return True
                if self._regen_btn.collidepoint(mx,my) and not self.sim_running:
                    self._start_regen(); return True
                if self._save_btn.collidepoint(mx,my):
                    save_settings(self.committed)
                    self.status_msg="Settings saved."; self.status_col=C_OK
                    self.needs_redraw=True; return True
            return False

        if ev.type==MOUSEBUTTONUP and ev.button==1:
            for sl in self.sliders: sl.dragging=False
            return False

        if ev.type==KEYDOWN:
            # Frame number input
            if self._frame_editing:
                if ev.key==K_RETURN:
                    try:
                        f=int(self._frame_buf)-1
                        self.cur_frame=max(0,min(self.player.count-1,f))
                    except ValueError: pass
                    self._frame_editing=False
                elif ev.key==K_BACKSPACE: self._frame_buf=self._frame_buf[:-1]
                elif ev.key==K_ESCAPE: self._frame_editing=False
                elif ev.unicode.isdigit(): self._frame_buf+=ev.unicode
                self.needs_redraw=True; return True
            # Slider text input
            for sl in self.sliders:
                if sl.editing:
                    if ev.key==K_RETURN:
                        try:
                            v=float(sl.edit_buf)
                            self.preview[sl.key]=max(sl.lo,min(sl.hi,v))
                            self.dirty=True
                        except ValueError: pass
                        sl.editing=False
                    elif ev.key==K_BACKSPACE: sl.edit_buf=sl.edit_buf[:-1]
                    elif ev.key==K_ESCAPE: sl.editing=False
                    elif ev.unicode in "0123456789.-": sl.edit_buf+=ev.unicode
                    self.needs_redraw=True; return True
        return False

    def _hit(self,mx,my):
        for n,r in [("prev",self._prev_btn),("next",self._next_btn),
                    ("play",self._play_btn),("regen",self._regen_btn),("save",self._save_btn)]:
            if r.collidepoint(mx,my): return n
        return None

    # ── simulation ────────────────────────────────────────────────────────────

    def _start_regen(self):
        saved_frame      = self.cur_frame
        self.committed   = copy.deepcopy(self.preview)
        self.dirty       = False
        self.sim_running = True
        self.status_msg  = "Simulation running..."
        self.status_col  = C_ACCENT
        self.needs_redraw= True
        threading.Thread(target=self._regen_worker,
                         args=(saved_frame,), daemon=True).start()

    def _regen_worker(self, saved_frame):
        s=self.committed; base=Path(__file__).parent
        cmd=[sys.executable,str(base/"simulate.py"),
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
             "--no-viewer"]
        try:
            r=subprocess.run(cmd,cwd=str(base),capture_output=True,text=True,timeout=600)
            if r.returncode==0:
                self.player.refresh()
                # Preserve frame index, clamped to new count
                self.cur_frame    = min(saved_frame, self.player.count-1)
                self.status_msg   = f"Done — {self.player.count} frames."
                self.status_col   = C_OK
            else:
                self.status_msg="Simulation failed — see console."; self.status_col=C_ERR
                print(r.stderr[-2000:])
        except Exception as e:
            self.status_msg=f"Error: {e}"; self.status_col=C_ERR
        finally:
            self.sim_running=False; self.needs_redraw=True


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(frames_dir="./frames"):
    pygame.init()
    pygame.display.set_caption("Stent FEA Viewer")
    pygame.display.set_mode((WIN_W,WIN_H), DOUBLEBUF|OPENGL)

    glClearColor(*[c/255. for c in C_BG],1.)
    glEnable(GL_NORMALIZE); setup_lighting()
    print(f"[viewer] GPU: {glGetString(GL_RENDERER).decode()}")

    settings = load_settings()
    player   = FramePlayer(frames_dir)
    stars    = StarField()
    camera   = Camera()
    panel    = UIPanel(PANEL_W, WIN_H, settings, player)
    panel.init()

    panel_tex = None
    clock     = pygame.time.Clock()
    t_prev    = time.time()

    while True:
        now = time.time(); dt = min(now-t_prev, .05); t_prev=now
        ctrl = bool(pygame.key.get_mods() & KMOD_CTRL)

        # Ensure anchors loaded (needs GL context, done lazily)
        player.ensure_ready()

        for ev in pygame.event.get():
            if ev.type==QUIT:                                     _quit(panel_tex,player); return
            if ev.type==KEYDOWN:
                if ev.key==K_ESCAPE:                              _quit(panel_tex,player); return
                if ev.key==K_LEFT:
                    panel.cur_frame=max(0,panel.cur_frame-1); panel.needs_redraw=True
                if ev.key==K_RIGHT:
                    panel.cur_frame=min(player.count-1,panel.cur_frame+1); panel.needs_redraw=True
                panel.handle(ev,VIEW_W)
            elif ev.type==MOUSEBUTTONDOWN:
                if ev.pos[0]<VIEW_W: camera.press(ev.pos[0],ev.pos[1],ctrl)
                else: panel.handle(ev,VIEW_W)
            elif ev.type==MOUSEBUTTONUP:
                camera.release(); panel.handle(ev,VIEW_W)
            elif ev.type==MOUSEMOTION:
                if ev.pos[0]<VIEW_W: camera.drag(ev.pos[0],ev.pos[1])
                else: panel.handle(ev,VIEW_W)
            elif ev.type==MOUSEWHEEL:
                if pygame.mouse.get_pos()[0]<VIEW_W: camera.scroll(ev.y)

        panel.tick(dt)
        stars.update(dt)

        # Update VBO with current frame + preview params
        player.update(panel.cur_frame, panel.preview, panel.committed, now)

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        # 1 — Stars
        glViewport(0,0,VIEW_W,WIN_H)
        glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-1,1,-1,1,-1,1)
        glMatrixMode(GL_MODELVIEW);  glLoadIdentity()
        stars.draw()

        # 2 — Stent
        glViewport(0,0,VIEW_W,WIN_H)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(42.,VIEW_W/WIN_H,.1,800.)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glTranslatef(0.,0.,-55.); camera.apply()
        if player.vbo:
            glScalef(player.scale,player.scale,player.scale)
            c=player.centre; glTranslatef(-float(c[0]),-float(c[1]),-float(c[2]))
            player.draw(preview=panel.dirty)

        # 3 — Panel texture (re-upload only when dirty)
        if panel.needs_redraw:
            surf=panel.render()
            if panel_tex: glDeleteTextures(1,[panel_tex])
            panel_tex=surf_to_tex(surf)
        if panel_tex: draw_panel_tex(panel_tex,VIEW_W,WIN_W,WIN_H)

        pygame.display.flip(); clock.tick(FPS)


def _quit(tex,player):
    if tex:
        try: glDeleteTextures(1,[tex])
        except Exception: pass
    player.free(); pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "./frames")
