import cv2
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

import mediapipe as mp
import numpy as np
import math
import time
import sys
try:
    import pygame
    from pygame.locals import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
    PYOPENGL_AVAILABLE = True
except ImportError:
    PYOPENGL_AVAILABLE = False
    print("Warning: PyOpenGL or Pygame not found. Falling back to 2D wireframe rendering.")

# --- CONSTANTS ---
WIN_W = 1280
WIN_H = 720
GL_W = 640          # render GL at half res for speed
GL_H = 360
ROTATION_SENSITIVITY = 80.0
TILT_SENSITIVITY = 300.0
SMOOTH_ROT = 0.18
SMOOTH_POS = 0.4
SMOOTH_SCALE = 0.35
PINCH_SCALE_FACTOR = 2.5   # single-hand pinch sensitivity

# --- SHAPE OPTIONS ---
SHAPES = ["cube", "sphere", "pyramid"]


class HandTracker:
    """Handles MediaPipe hand tracking and gesture recognition."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hand_data = []
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                pts = []
                for lm in hand_lm.landmark:
                    pts.append((int(lm.x * WIN_W), int(lm.y * WIN_H), lm.z))

                thumb_ext  = math.hypot(pts[4][0] - pts[2][0], pts[4][1] - pts[2][1]) > 35
                index_up   = pts[8][1]  < pts[6][1]
                middle_up  = pts[12][1] < pts[10][1]
                ring_up    = pts[16][1] < pts[14][1]
                pinky_up   = pts[20][1] < pts[18][1]
                ring_folded  = not ring_up
                pinky_folded = not pinky_up

                # Thumb-index pinch distance (normalised to frame diagonal)
                pinch_dist = math.hypot(pts[4][0] - pts[8][0], pts[4][1] - pts[8][1])

                # 3-finger active gesture (thumb+index+middle)
                is_active = thumb_ext and index_up and middle_up and ring_folded and pinky_folded

                # Peace sign (index+middle only, no thumb)
                is_peace = (not thumb_ext) and index_up and middle_up and ring_folded and pinky_folded

                # Open palm (all fingers up)
                is_open_palm = index_up and middle_up and ring_up and pinky_up

                # Fist (all fingers folded)
                is_fist = (not index_up) and (not middle_up) and (not ring_up) and (not pinky_up)

                hand_data.append({
                    'pts': pts,
                    'is_active': is_active,
                    'is_peace': is_peace,
                    'is_open_palm': is_open_palm,
                    'is_fist': is_fist,
                    'pinch_dist': pinch_dist,
                })

        return results, hand_data

    def draw_landmarks(self, frame, results):
        """Draws white skeleton on hands."""
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                connections = self.mp_hands.HAND_CONNECTIONS
                for connection in connections:
                    idx1, idx2 = connection
                    lm1 = hand_lm.landmark[idx1]
                    lm2 = hand_lm.landmark[idx2]
                    x1, y1 = int(lm1.x * WIN_W), int(lm1.y * WIN_H)
                    x2, y2 = int(lm2.x * WIN_W), int(lm2.y * WIN_H)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                for lm in hand_lm.landmark:
                    x, y = int(lm.x * WIN_W), int(lm.y * WIN_H)
                    cv2.circle(frame, (x, y), 3,  (255, 255, 255), -1)
                    cv2.circle(frame, (x, y), 8,  (255, 255, 255), 1)
                    cv2.circle(frame, (x, y), 12, (255, 255, 255), 1)


class AR3DObject:
    """Manages the 3D PyOpenGL object state and rendering."""

    FACE_COLORS = [
        (1.0, 0.2, 0.2, 0.55),   # red
        (0.2, 0.9, 0.3, 0.55),   # green
        (0.2, 0.4, 1.0, 0.55),   # blue
        (1.0, 0.85, 0.1, 0.55),  # yellow
        (0.9, 0.2, 0.9, 0.55),   # magenta
        (0.1, 0.9, 0.9, 0.55),   # cyan
    ]

    def __init__(self):
        self.pos          = [0.0, 0.0, -5.0]
        self.target_pos   = [0.0, 0.0, -5.0]
        self.rot_x        = 0.0
        self.rot_y        = 0.0
        self.rot_z        = 0.0
        self.target_rot_x = 0.0
        self.target_rot_y = 0.0
        self.target_rot_z = 0.0
        self.scale        = 1.0
        self.target_scale = 1.0
        self.visible      = False
        self.frozen       = False   # open palm freezes rotation
        self.shape_index  = 0      # 0=cube, 1=sphere, 2=pyramid

        if PYOPENGL_AVAILABLE:
            self._init_gl()
            self._quad = gluNewQuadric()
            gluQuadricDrawStyle(self._quad, GLU_LINE)

    def _init_gl(self):
        pygame.init()
        pygame.display.set_mode((GL_W, GL_H), DOUBLEBUF | OPENGL | pygame.HIDDEN)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def update(self):
        """Smoothly interpolate rotation, position and scale."""
        if not self.frozen:
            self.rot_x += (self.target_rot_x - self.rot_x) * SMOOTH_ROT
            self.rot_y += (self.target_rot_y - self.rot_y) * SMOOTH_ROT
            self.rot_z += (self.target_rot_z - self.rot_z) * SMOOTH_ROT

        for i in range(3):
            self.pos[i] += (self.target_pos[i] - self.pos[i]) * SMOOTH_POS

        self.scale += (self.target_scale - self.scale) * SMOOTH_SCALE

    def set_pos_from_screen(self, screen_x, screen_y):
        ratio_x = WIN_W / WIN_H
        gl_x = ((screen_x / WIN_W) * 2.0 - 1.0) * ratio_x * 2.5
        gl_y = -((screen_y / WIN_H) * 2.0 - 1.0) * 2.5
        self.target_pos[0] = gl_x
        self.target_pos[1] = gl_y

    def cycle_shape(self):
        self.shape_index = (self.shape_index + 1) % len(SHAPES)

    def _draw_cube(self):
        vertices = (
            (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
            (1, -1,  1), (1, 1,  1), (-1, -1,  1), (-1, 1,  1)
        )
        edges = (
            (0,1),(0,3),(0,4),(2,1),(2,3),(2,7),
            (6,3),(6,4),(6,7),(5,1),(5,4),(5,7)
        )
        faces = (
            (0,1,2,3),(3,2,7,6),(6,7,5,4),
            (4,5,1,0),(1,5,7,2),(4,0,3,6)
        )

        # Colored faces
        glBegin(GL_QUADS)
        for i, face in enumerate(faces):
            glColor4f(*self.FACE_COLORS[i])
            for v in face:
                glVertex3fv(vertices[v])
        glEnd()

        # Pulsing white edges
        pulse = (math.sin(time.time() * 3) + 1) / 2
        glLineWidth(3)
        glBegin(GL_LINES)
        glColor3f(0.6 + 0.4 * pulse, 0.8 + 0.2 * pulse, 1.0)
        for edge in edges:
            for v in edge:
                glVertex3fv(vertices[v])
        glEnd()

    def _draw_sphere(self):
        pulse = (math.sin(time.time() * 3) + 1) / 2
        glColor4f(0.2, 0.6 + 0.4 * pulse, 1.0, 0.8)
        gluSphere(self._quad, 1.0, 20, 20)

    def _draw_pyramid(self):
        apex = (0.0, 1.5, 0.0)
        base = [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]

        # Colored triangle faces
        glBegin(GL_TRIANGLES)
        for i in range(4):
            glColor4f(*self.FACE_COLORS[i])
            glVertex3fv(base[i])
            glVertex3fv(base[(i + 1) % 4])
            glVertex3fv(apex)
        glEnd()

        # Base quad
        glBegin(GL_QUADS)
        glColor4f(*self.FACE_COLORS[4])
        for v in base:
            glVertex3fv(v)
        glEnd()

        # Edges
        pulse = (math.sin(time.time() * 3) + 1) / 2
        glLineWidth(3)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.8 + 0.2 * pulse, 0.2)
        for i in range(4):
            glVertex3fv(base[i]); glVertex3fv(base[(i + 1) % 4])
            glVertex3fv(base[i]); glVertex3fv(apex)
        glEnd()

    def render_gl(self):
        if not self.visible or not PYOPENGL_AVAILABLE:
            return None

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluPerspective(45, GL_W / GL_H, 0.1, 50.0)
        glTranslate(*self.pos)
        glRotate(self.rot_x, 1, 0, 0)
        glRotate(self.rot_y, 0, 1, 0)
        glRotate(self.rot_z, 0, 0, 1)
        glScale(self.scale, self.scale, self.scale)

        shape = SHAPES[self.shape_index]
        if shape == "cube":
            self._draw_cube()
        elif shape == "sphere":
            self._draw_sphere()
        elif shape == "pyramid":
            self._draw_pyramid()

        pygame.display.flip()
        # Read at the smaller GL resolution then upscale — big perf win
        pixels = glReadPixels(0, 0, GL_W, GL_H, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(GL_H, GL_W, 3)
        image = np.flipud(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if GL_W != WIN_W or GL_H != WIN_H:
            image = cv2.resize(image, (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)
        return image

    def render_2d_fallback(self, frame):
        if not self.visible:
            return frame
        cx = int(WIN_W / 2 + self.pos[0] * 100)
        cy = int(WIN_H / 2 - self.pos[1] * 100)
        s  = int(50 * self.scale)
        cv2.rectangle(frame, (cx - s, cy - s), (cx + s, cy + s), (0, 255, 255), 2)
        return frame


class ARApplication:
    """Main application orchestrating tracking and 3D rendering."""

    def __init__(self):
        self.cap             = None
        self.tracker         = HandTracker()
        self.obj             = AR3DObject()
        self.prev_roll_angle = None
        self.prev_tilt       = None
        self.prev_z          = None
        self.prev_peace      = False
        self.prev_pinch      = None   # previous pinch distance for single-hand scale
        self.last_t          = time.time()
        # Cooldown so one peace-sign doesn't spam shape changes
        self.peace_cooldown  = 0.0

    def initialize(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIN_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)
        print("AR Hand Studio - Improved Edition")
        print("Gestures:")
        print("  3-finger (thumb+index+middle) : show & rotate object")
        print("  Pinch (thumb+index)            : scale object")
        print("  Peace sign (index+middle)      : cycle shape (cube -> sphere -> pyramid)")
        print("  Open palm                      : freeze / unfreeze rotation")
        print("  Fist                           : hide object")
        print("  Two hands                      : pinch to scale")
        print("Press ESC to quit.")

    def _draw_hud(self, frame, hand_data):
        """Overlay FPS, active gesture label and current shape."""
        now = time.time()
        fps = 1.0 / max(now - self.last_t, 1e-9)
        self.last_t = now

        shape_name = SHAPES[self.obj.shape_index].upper()
        frozen_tag = "  [FROZEN]" if self.obj.frozen else ""

        cv2.putText(frame, f"FPS: {fps:.0f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
        cv2.putText(frame, f"Shape: {shape_name}{frozen_tag}",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        cv2.putText(frame, f"Scale: {self.obj.scale:.2f}x",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        if hand_data:
            h = hand_data[0]
            if h['is_fist']:
                label = "FIST — hidden"
            elif h['is_open_palm']:
                label = "OPEN PALM — freeze"
            elif h['is_peace']:
                label = "PEACE — cycle shape"
            elif h['is_active']:
                label = "3-FINGER — rotate"
            else:
                label = "Tracking…"
            cv2.putText(frame, label, (20, WIN_H - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

        # Two-hand scale hint
        if len(hand_data) == 2:
            cv2.putText(frame, "PINCH — scaling",
                        (WIN_W - 280, WIN_H - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 180, 0), 2)

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            results, hand_data = self.tracker.process_frame(frame)
            self.tracker.draw_landmarks(frame, results)
            self._handle_gestures(hand_data)
            self.obj.update()

            # Render 3D object
            if PYOPENGL_AVAILABLE and self.obj.visible:
                gl_image = self.obj.render_gl()
                if gl_image is not None:
                    gray = cv2.cvtColor(gl_image, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                    frame = np.where(mask[:, :, None] == 255, gl_image, frame)
            elif not PYOPENGL_AVAILABLE and self.obj.visible:
                frame = self.obj.render_2d_fallback(frame)

            self._draw_hud(frame, hand_data)
            cv2.imshow("AR Hand Studio — Improved", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cleanup()

    def _handle_gestures(self, hand_data):
        now = time.time()

        # --- Two-hand pinch: scale ---
        if len(hand_data) == 2:
            tip1 = hand_data[0]['pts'][8]
            tip2 = hand_data[1]['pts'][8]
            dist = math.hypot(tip1[0] - tip2[0], tip1[1] - tip2[1])
            # Map ~50 px → 0.4x,  ~400 px → 3.0x
            self.obj.target_scale = max(0.4, min(3.0, dist / 130.0))
            self.obj.visible = True  # keep object visible while scaling
            return

        # Default: hide object
        self.obj.visible = False
        in_rotate_mode   = False

        if len(hand_data) == 0:
            self.prev_roll_angle = None
            self.prev_tilt       = None
            self.prev_z          = None
            self.prev_pinch      = None
            return

        hand = hand_data[0]
        pts  = hand['pts']

        # FIST — hide
        if hand['is_fist']:
            self.obj.visible = False
            self.prev_roll_angle = None
            self.prev_tilt       = None
            self.prev_z          = None
            self.prev_peace      = False
            self.prev_pinch      = None
            return

        # OPEN PALM — freeze / show
        if hand['is_open_palm']:
            self.obj.visible = True
            self.obj.frozen  = True
            self.prev_roll_angle = None
            self.prev_tilt       = None
            self.prev_z          = None
            self.prev_peace      = False
            return

        # PEACE — cycle shape (rising-edge only, with cooldown)
        if hand['is_peace']:
            self.obj.visible = True
            self.obj.frozen  = False
            if not self.prev_peace and now > self.peace_cooldown:
                self.obj.cycle_shape()
                self.peace_cooldown = now + 0.6
            self.prev_peace = True
            self.prev_roll_angle = None
            self.prev_tilt       = None
            self.prev_z          = None
            return
        self.prev_peace = False

        # 3-FINGER — show & rotate + single-hand pinch to scale
        if hand['is_active']:
            self.obj.visible = True
            self.obj.frozen  = False
            in_rotate_mode   = True

            # Position above the three fingertips
            cx    = (pts[4][0] + pts[8][0] + pts[12][0]) // 3
            min_y = min(pts[4][1], pts[8][1], pts[12][1])
            cy    = min_y - 120
            self.obj.set_pos_from_screen(cx, cy)

            # --- Single-hand pinch scale (thumb↔index distance) ---
            pinch = hand['pinch_dist']
            if self.prev_pinch is not None:
                delta_pinch = pinch - self.prev_pinch
                self.obj.target_scale += delta_pinch / 100.0 * PINCH_SCALE_FACTOR
                self.obj.target_scale = max(0.3, min(4.0, self.obj.target_scale))
            self.prev_pinch = pinch

            # Y-axis rotation — wrist roll
            roll_angle = math.atan2(pts[17][1] - pts[5][1],
                                    pts[17][0] - pts[5][0])
            if self.prev_roll_angle is not None:
                delta_roll = roll_angle - self.prev_roll_angle
                self.obj.target_rot_y += delta_roll * ROTATION_SENSITIVITY
            self.prev_roll_angle = roll_angle

            # X-axis rotation — wrist tilt (depth)
            tilt = pts[0][2] - pts[9][2]
            if self.prev_tilt is not None:
                delta_tilt = tilt - self.prev_tilt
                self.obj.target_rot_x += delta_tilt * TILT_SENSITIVITY
            self.prev_tilt = tilt

            # Clamp X so the object doesn't flip upside-down
            self.obj.target_rot_x = max(-90.0, min(90.0, self.obj.target_rot_x))

            # Z-axis rotation — wrist spread depth
            depth_z = pts[0][2] - pts[17][2]
            if self.prev_z is not None:
                delta_z = depth_z - self.prev_z
                self.obj.target_rot_z += delta_z * 200.0
            self.prev_z = depth_z

        if not in_rotate_mode:
            self.prev_roll_angle = None
            self.prev_tilt       = None
            self.prev_z          = None
            self.prev_pinch      = None

    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if PYOPENGL_AVAILABLE:
            pygame.quit()


if __name__ == "__main__":
    app = ARApplication()
    app.initialize()
    app.run()