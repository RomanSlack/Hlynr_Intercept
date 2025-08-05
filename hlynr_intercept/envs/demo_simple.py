#!/usr/bin/env python3
"""
Minimal 3‑D missile‑intercept visualiser.
No ModernGL‑legacy mixing, no external resources.
"""

import time, math, sys
import numpy as np
import pyglet
from pyglet.window import key, mouse
import moderngl as mgl
from pyrr import Matrix44, Vector3, Quaternion

# ───────────────────────────── utility ──────────────────────────────
def unit(v):              # safe normalise
    n = np.linalg.norm(v)
    return v / n if n else v

def euler_to_quat(rpy):    # roll, pitch, yaw → quaternion
    return Quaternion.from_eulers([rpy[0], rpy[1], rpy[2]])

def quat_to_mat4(q):       # quaternion → 4×4 rotation matrix
    m = Matrix44.identity(dtype='f4')
    m[:3,:3] = q.matrix33
    return m

# ───────────────────── simple physics objects ───────────────────────
class Missile:
    def __init__(self, colour, length, pos, vel, rpy):
        self.colour   = colour
        self.length   = length
        self.pos      = Vector3(pos, dtype='f4')
        self.vel      = Vector3(vel, dtype='f4')
        self.rpy      = np.array(rpy, dtype='f4')

    def step(self, dt):
        self.pos += self.vel * dt                                   # straight‑line for demo
        self.rpy[2] += 0.2*dt                                        # yaw slowly so you see rotation

# ───────────────────────── main window ──────────────────────────────
class InterceptWindow(pyglet.window.Window):

    def __init__(self):
        config = pyglet.gl.Config(double_buffer=True, depth_size=24, samples=4)
        super().__init__(1400, 900, "Hlynr minimal demo", resizable=True, config=config)
        self.ctx = mgl.create_context(require=330)
        self.ctx.enable(mgl.DEPTH_TEST)
        self._build_programs()
        self._build_geometry()
        self._init_scene()
        pyglet.clock.schedule_interval(self.update, 1/60)

    # ── GL resources ────────────────────────────────────────────────
    def _build_programs(self):
        VERT = '''
        #version 330
        in vec3 in_pos;
        in vec3 in_nrm;
        uniform mat4 mvp;
        uniform mat4 model;
        out vec3 nrm_w;
        out vec3 pos_w;
        void main(){
            pos_w = (model*vec4(in_pos,1)).xyz;
            nrm_w = normalize((model*vec4(in_nrm,0)).xyz);
            gl_Position = mvp*vec4(in_pos,1);
        }'''
        FRAG = '''
        #version 330
        in vec3 nrm_w;
        in vec3 pos_w;
        uniform vec3 colour;
        uniform vec3 light_dir;         // world space dir, normalised
        uniform vec3 cam_pos;
        out vec4 frag;
        void main(){
            vec3 N = normalize(nrm_w);
            vec3 L = normalize(-light_dir);
            float diff = max(dot(N,L),0);
            vec3 V = normalize(cam_pos - pos_w);
            vec3 R = reflect(-L,N);
            float spec = pow(max(dot(V,R),0),32);
            vec3 ambient = 0.15*colour;
            vec3 result = ambient + diff*colour + spec*vec3(1);
            frag = vec4(result,1);
        }'''
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)

    def _build_geometry(self):
        # ground plane (2 triangles)
        plane = np.array([
            #  x,   y, z,   nx, ny, nz
            -5000, -5000, 0, 0,0,1,
             5000, -5000, 0, 0,0,1,
             5000,  5000, 0, 0,0,1,
            -5000,  5000, 0, 0,0,1,
        ], dtype='f4')
        idx = np.array([0,1,2, 2,3,0], dtype='i4')
        vbo = self.ctx.buffer(plane.tobytes())
        ibo = self.ctx.buffer(idx.tobytes())
        self.vao_plane = self.ctx.vertex_array(
            self.prog,
            [(vbo, '3f 3f', 'in_pos','in_nrm')],
            ibo)

        # missile body = cylinder+cone approximated by 16‑segment prism
        seg = 16
        verts = []; norms = []; idx = []
        r = 0.05                                     # 0.05 m radius (before scaling)
        for i in range(seg):
            a0 = 2*math.pi*i/seg
            a1 = 2*math.pi*(i+1)/seg
            x0,y0 = r*math.cos(a0), r*math.sin(a0)
            x1,y1 = r*math.cos(a1), r*math.sin(a1)
            # quad side (two tris) from z=0 to z=0.8
            verts += [[x0,y0,0],[x0,y0,0.8],[x1,y1,0.8],[x1,y1,0]]
            n = unit(np.array([x0, y0, 0]))
            norms += [n]*4
            base = 4*i
            idx += [base,base+1,base+2, base+2,base+3,base]
        # cone tip
        verts += [[0,0,1]]
        norms += [[0,0,1]]
        tip = len(verts)-1
        for i in range(seg):
            idx += [tip, (4*i+1), (4*((i+1)%seg)+1)]

        verts = np.array(verts, dtype='f4')
        norms = np.array(norms, dtype='f4')
        idx   = np.array(idx,   dtype='i4')
        vbo = self.ctx.buffer(np.hstack([verts, norms]).tobytes())
        ibo = self.ctx.buffer(idx.tobytes())
        self.vao_missile = self.ctx.vertex_array(
            self.prog,
            [(vbo, '3f 3f', 'in_pos','in_nrm')],
            ibo)

    # ── scene setup ─────────────────────────────────────────────────
    def _init_scene(self):
        self.cam_pos   = Vector3([0,-300,100], dtype='f4')
        self.cam_yaw   = 0.0          # radians
        self.cam_pitch = math.radians(15)
        self.light_dir = unit(np.array([ 0.3,-0.4,-1]))

        self.interceptor = Missile([0.2,0.2,1], 3, [0,0,50],   [0,0,0],  [0,0,0])
        self.attacker    = Missile([1,0.2,0.2], 4, [200,0,100], [-20,0,-10], [0,0,0])

        self.keys = set()

    # ── camera helpers ──────────────────────────────────────────────
    def _update_camera(self, dt):
        SPEED = 2500*dt
        yaw   = self.cam_yaw
        pitch = self.cam_pitch
        fwd = Vector3([ math.sin(yaw)*math.cos(pitch),
                        math.cos(yaw)*math.cos(pitch),
                        math.sin(pitch)])
        right = Vector3([ math.cos(yaw), -math.sin(yaw), 0])
        up    = Vector3([0,0,1])
        if key.W in self.keys: self.cam_pos += fwd*SPEED
        if key.S in self.keys: self.cam_pos -= fwd*SPEED
        if key.A in self.keys: self.cam_pos -= right*SPEED
        if key.D in self.keys: self.cam_pos += right*SPEED
        if key.Q in self.keys: self.cam_pos += up*SPEED
        if key.E in self.keys: self.cam_pos -= up*SPEED

    def on_mouse_drag(self,x,y,dx,dy,buttons,mods):
        if buttons & mouse.RIGHT:
            self.cam_yaw   += dx*0.005
            self.cam_pitch = np.clip(self.cam_pitch + dy*0.005, -1.2, 1.2)

    def on_key_press(self,sym,mods):
        self.keys.add(sym)
        if sym==key.ESCAPE: pyglet.app.exit()

    def on_key_release(self,sym,mods):
        self.keys.discard(sym)

    # ── main update/render ──────────────────────────────────────────
    def update(self, dt):
        self._update_camera(dt)
        self.interceptor.step(dt)
        self.attacker.step(dt)
        self.render()

    def render(self):
        self.ctx.clear(0.05,0.08,0.12)
        self.ctx.enable(mgl.DEPTH_TEST)

        # matrices
        proj = Matrix44.perspective_projection(60, self.width/self.height, 10, 50000, dtype='f4')
        target = self.cam_pos + Vector3([ math.sin(self.cam_yaw)*math.cos(self.cam_pitch),
                                          math.cos(self.cam_yaw)*math.cos(self.cam_pitch),
                                          math.sin(self.cam_pitch)])
        view = Matrix44.look_at(self.cam_pos, target, Vector3([0,0,1]), dtype='f4')

        # draw ground
        m_model = Matrix44.identity(dtype='f4')
        self._draw(self.vao_plane, m_model, view, proj, np.array([0.3,0.6,0.3]))

        # draw missiles
        self._draw_missile(self.interceptor, view, proj)
        self._draw_missile(self.attacker,    view, proj)

    # ── helpers ─────────────────────────────────────────────────────
    def _draw(self, vao, model, view, proj, colour):
        self.prog['mvp'].write((proj*view*model).astype('f4').tobytes())
        self.prog['model'].write(model.astype('f4').tobytes())
        self.prog['colour'].value = tuple(colour)
        self.prog['light_dir'].value = tuple(self.light_dir)
        self.prog['cam_pos'].value  = tuple(self.cam_pos)
        vao.render()

    def _draw_missile(self, msl, view, proj):
        # scale + rotate + translate
        S = Matrix44.from_scale([msl.length*0.5, msl.length*0.5, msl.length*5], dtype='f4')
        R = quat_to_mat4(euler_to_quat(msl.rpy))
        T = Matrix44.from_translation(msl.pos, dtype='f4')
        model = T*R*S
        self._draw(self.vao_missile, model, view, proj, msl.colour)

# ────────────────────────── run app ─────────────────────────────────
if __name__ == '__main__':
    try:
        win = InterceptWindow()
        pyglet.app.run()
    except Exception as e:
        print("Fatal:", e)
        sys.exit(1)
