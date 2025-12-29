"""
3D Model Viewer with Hand Gesture Control
==========================================
Supports loading and manipulating 3D models with hand gestures.
Uses PyOpenGL for rendering and supports OBJ, STL, and PLY formats.
"""

import sys
import os
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QSurfaceFormat

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("PyOpenGL not available. 3D viewer will be disabled.")

import struct
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Mesh:
    """Simple mesh data structure"""
    vertices: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    colors: Optional[np.ndarray] = None
    
    @property
    def center(self) -> np.ndarray:
        return np.mean(self.vertices, axis=0)
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)
    
    @property
    def scale(self) -> float:
        min_b, max_b = self.bounds
        return np.max(max_b - min_b)


class ModelLoader:
    """Load various 3D model formats"""
    
    @staticmethod
    def load(filepath: str) -> Optional[Mesh]:
        """Load a 3D model file"""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.obj':
            return ModelLoader.load_obj(filepath)
        elif ext == '.stl':
            return ModelLoader.load_stl(filepath)
        elif ext == '.ply':
            return ModelLoader.load_ply(filepath)
        else:
            print(f"Unsupported format: {ext}")
            return None
    
    @staticmethod
    def load_obj(filepath: str) -> Optional[Mesh]:
        """Load OBJ file"""
        vertices = []
        normals = []
        faces = []
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'v':
                        vertices.append([float(x) for x in parts[1:4]])
                    elif parts[0] == 'vn':
                        normals.append([float(x) for x in parts[1:4]])
                    elif parts[0] == 'f':
                        face = []
                        for p in parts[1:]:
                            indices = p.split('/')
                            face.append(int(indices[0]) - 1)
                        faces.append(face[:3])  # Triangles only
            
            vertices = np.array(vertices, dtype=np.float32)
            
            if normals:
                normals = np.array(normals, dtype=np.float32)
            else:
                normals = ModelLoader.calculate_normals(vertices, np.array(faces))
            
            return Mesh(
                vertices=vertices,
                normals=normals,
                faces=np.array(faces, dtype=np.uint32)
            )
        except Exception as e:
            print(f"Error loading OBJ: {e}")
            return None
    
    @staticmethod
    def load_stl(filepath: str) -> Optional[Mesh]:
        """Load STL file (binary or ASCII)"""
        try:
            with open(filepath, 'rb') as f:
                header = f.read(80)
                
                if b'solid' in header[:5]:
                    f.seek(0)
                    return ModelLoader.load_stl_ascii(f)
                else:
                    return ModelLoader.load_stl_binary(f)
        except Exception as e:
            print(f"Error loading STL: {e}")
            return None
    
    @staticmethod
    def load_stl_binary(f) -> Mesh:
        """Load binary STL"""
        num_triangles = struct.unpack('<I', f.read(4))[0]
        
        vertices = []
        normals = []
        faces = []
        
        for i in range(num_triangles):
            normal = struct.unpack('<3f', f.read(12))
            normals.extend([normal] * 3)
            
            for _ in range(3):
                vertex = struct.unpack('<3f', f.read(12))
                vertices.append(vertex)
            
            faces.append([i*3, i*3+1, i*3+2])
            f.read(2)
        
        return Mesh(
            vertices=np.array(vertices, dtype=np.float32),
            normals=np.array(normals, dtype=np.float32),
            faces=np.array(faces, dtype=np.uint32)
        )
    
    @staticmethod
    def load_stl_ascii(f) -> Mesh:
        """Load ASCII STL"""
        vertices = []
        normals = []
        faces = []
        current_normal = None
        vertex_count = 0
        
        for line in f:
            line = line.decode('utf-8', errors='ignore').strip()
            parts = line.split()
            
            if not parts:
                continue
            
            if parts[0] == 'facet' and parts[1] == 'normal':
                current_normal = [float(x) for x in parts[2:5]]
            elif parts[0] == 'vertex':
                vertices.append([float(x) for x in parts[1:4]])
                normals.append(current_normal)
                vertex_count += 1
                
                if vertex_count % 3 == 0:
                    idx = vertex_count - 3
                    faces.append([idx, idx+1, idx+2])
        
        return Mesh(
            vertices=np.array(vertices, dtype=np.float32),
            normals=np.array(normals, dtype=np.float32),
            faces=np.array(faces, dtype=np.uint32)
        )
    
    @staticmethod
    def load_ply(filepath: str) -> Optional[Mesh]:
        """Load PLY file"""
        try:
            vertices = []
            faces = []
            colors = []
            
            with open(filepath, 'rb') as f:
                vertex_count = 0
                face_count = 0
                has_color = False
                header_end = False
                is_binary = False
                
                while not header_end:
                    line = f.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line.startswith('format'):
                        is_binary = 'binary' in line
                    elif line.startswith('element vertex'):
                        vertex_count = int(line.split()[-1])
                    elif line.startswith('element face'):
                        face_count = int(line.split()[-1])
                    elif 'red' in line or 'green' in line or 'blue' in line:
                        has_color = True
                    elif line == 'end_header':
                        header_end = True
                
                if not is_binary:
                    for _ in range(vertex_count):
                        line = f.readline().decode('utf-8', errors='ignore').strip()
                        parts = line.split()
                        vertices.append([float(x) for x in parts[:3]])
                        if has_color and len(parts) >= 6:
                            colors.append([int(x)/255.0 for x in parts[3:6]])
                    
                    for _ in range(face_count):
                        line = f.readline().decode('utf-8', errors='ignore').strip()
                        parts = line.split()
                        n = int(parts[0])
                        face = [int(x) for x in parts[1:n+1]]
                        faces.append(face[:3])
            
            vertices = np.array(vertices, dtype=np.float32)
            normals = ModelLoader.calculate_normals(vertices, np.array(faces))
            
            return Mesh(
                vertices=vertices,
                normals=normals,
                faces=np.array(faces, dtype=np.uint32),
                colors=np.array(colors, dtype=np.float32) if colors else None
            )
        except Exception as e:
            print(f"Error loading PLY: {e}")
            return None
    
    @staticmethod
    def calculate_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Calculate vertex normals from faces"""
        normals = np.zeros_like(vertices)
        
        for face in faces:
            if len(face) >= 3:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal /= norm
                
                for idx in face[:3]:
                    normals[idx] += normal
        
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normals /= norms
        
        return normals.astype(np.float32)


class ModelViewer(QOpenGLWidget):
    """OpenGL widget for 3D model viewing with gesture control"""
    
    def __init__(self, parent=None):
        if not OPENGL_AVAILABLE:
            super().__init__(parent)
            return
            
        fmt = QSurfaceFormat()
        fmt.setSamples(4)
        QSurfaceFormat.setDefaultFormat(fmt)
        
        super().__init__(parent)
        
        self.setWindowTitle("3D Model Viewer")
        self.setMinimumSize(800, 600)
        
        self.mesh: Optional[Mesh] = None
        
        self.rotation_x = 30.0
        self.rotation_y = 45.0
        self.rotation_z = 0.0
        self.translation_x = 0.0
        self.translation_y = 0.0
        self.zoom = 3.0
        
        self.auto_rotate = False
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animate)
        self.animation_timer.start(16)
        
        self.last_pos = None
        self.current_button = None
        
        self.gesture_rotation_x = 0.0
        self.gesture_rotation_y = 0.0
        self.gesture_scale = 1.0
        self.gesture_trans_x = 0.0
        self.gesture_trans_y = 0.0
        
    def load_model(self, filepath: str):
        """Load a 3D model from file"""
        if not OPENGL_AVAILABLE:
            print("OpenGL not available")
            return
            
        self.mesh = ModelLoader.load(filepath)
        
        if self.mesh is not None:
            center = self.mesh.center
            scale = self.mesh.scale
            
            self.mesh.vertices = (self.mesh.vertices - center) / scale
            
            self.rotation_x = 30.0
            self.rotation_y = 45.0
            self.zoom = 2.0
            
            self.update()
            print(f"Loaded model with {len(self.mesh.vertices)} vertices")
        else:
            print(f"Failed to load model: {filepath}")
    
    def initializeGL(self):
        """Initialize OpenGL"""
        if not OPENGL_AVAILABLE:
            return
            
        glClearColor(0.1, 0.1, 0.15, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 10.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.6, 0.6, 0.8, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)
        
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)
        
    def resizeGL(self, width, height):
        """Handle resize"""
        if not OPENGL_AVAILABLE:
            return
            
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / height if height > 0 else 1.0
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
    def paintGL(self):
        """Render the scene"""
        if not OPENGL_AVAILABLE:
            return
            
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        glTranslatef(self.translation_x, self.translation_y, -self.zoom)
        
        glRotatef(self.rotation_x + self.gesture_rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.rotation_y + self.gesture_rotation_y, 0.0, 1.0, 0.0)
        glRotatef(self.rotation_z, 0.0, 0.0, 1.0)
        
        scale = self.gesture_scale
        glScalef(scale, scale, scale)
        
        if self.mesh is not None:
            self.draw_mesh()
        else:
            self.draw_default_cube()
        
        self.draw_grid()
        
    def draw_mesh(self):
        """Draw the loaded mesh"""
        glColor3f(0.6, 0.6, 0.8)
        
        glBegin(GL_TRIANGLES)
        for face in self.mesh.faces:
            for idx in face[:3]:
                if idx < len(self.mesh.normals):
                    glNormal3fv(self.mesh.normals[idx])
                if self.mesh.colors is not None and idx < len(self.mesh.colors):
                    glColor3fv(self.mesh.colors[idx])
                glVertex3fv(self.mesh.vertices[idx])
        glEnd()
        
    def draw_default_cube(self):
        """Draw a default cube when no model is loaded"""
        glColor3f(0.4, 0.6, 0.9)
        
        vertices = [
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
        ]
        
        faces = [
            (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
            (2, 6, 7, 3), (0, 3, 7, 4), (1, 5, 6, 2)
        ]
        
        normals = [
            (0, 0, -1), (0, 0, 1), (0, -1, 0),
            (0, 1, 0), (-1, 0, 0), (1, 0, 0)
        ]
        
        glBegin(GL_QUADS)
        for i, face in enumerate(faces):
            glNormal3fv(normals[i])
            for idx in face:
                glVertex3fv(vertices[idx])
        glEnd()
        
    def draw_grid(self):
        """Draw a reference grid"""
        glDisable(GL_LIGHTING)
        glColor3f(0.3, 0.3, 0.3)
        
        glBegin(GL_LINES)
        for i in range(-5, 6):
            glVertex3f(i * 0.2, -0.5, -1.0)
            glVertex3f(i * 0.2, -0.5, 1.0)
            glVertex3f(-1.0, -0.5, i * 0.2)
            glVertex3f(1.0, -0.5, i * 0.2)
        glEnd()
        
        glEnable(GL_LIGHTING)
        
    def animate(self):
        """Animation tick"""
        if self.auto_rotate:
            self.rotation_y += 0.5
            self.update()
        
        self.gesture_rotation_x *= 0.9
        self.gesture_rotation_y *= 0.9
        self.gesture_scale = 0.9 * self.gesture_scale + 0.1 * 1.0
        
        if abs(self.gesture_rotation_x) > 0.1 or abs(self.gesture_rotation_y) > 0.1:
            self.update()
    
    def rotate(self, delta_x: float, delta_y: float):
        """Apply rotation from gesture"""
        self.gesture_rotation_y += delta_x * 0.5
        self.gesture_rotation_x += delta_y * 0.5
        
        self.rotation_y += self.gesture_rotation_y * 0.1
        self.rotation_x += self.gesture_rotation_x * 0.1
        
        self.update()
    
    def scale(self, factor: float):
        """Apply scale from gesture"""
        self.gesture_scale *= factor
        self.gesture_scale = max(0.1, min(5.0, self.gesture_scale))
        self.update()
    
    def translate(self, delta_x: float, delta_y: float):
        """Apply translation from gesture"""
        self.translation_x += delta_x
        self.translation_y -= delta_y
        self.update()
    
    def reset_view(self):
        """Reset view to default"""
        self.rotation_x = 30.0
        self.rotation_y = 45.0
        self.rotation_z = 0.0
        self.translation_x = 0.0
        self.translation_y = 0.0
        self.zoom = 2.0
        self.gesture_scale = 1.0
        self.update()
    
    def mousePressEvent(self, event):
        self.last_pos = event.pos()
        self.current_button = event.button()
        
    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return
            
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        
        if self.current_button == Qt.LeftButton:
            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
        elif self.current_button == Qt.RightButton:
            self.translation_x += dx * 0.01
            self.translation_y -= dy * 0.01
        elif self.current_button == Qt.MiddleButton:
            self.zoom -= dy * 0.01
            self.zoom = max(0.5, min(10.0, self.zoom))
        
        self.last_pos = event.pos()
        self.update()
        
    def mouseReleaseEvent(self, event):
        self.last_pos = None
        self.current_button = None
        
    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120.0
        self.zoom -= delta * 0.2
        self.zoom = max(0.5, min(10.0, self.zoom))
        self.update()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            self.reset_view()
        elif event.key() == Qt.Key_A:
            self.auto_rotate = not self.auto_rotate
        elif event.key() == Qt.Key_Escape:
            self.close()


class ModelViewerWindow(QtWidgets.QMainWindow):
    """Standalone window for the 3D model viewer"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("3D Model Viewer - Gesture Control")
        self.setMinimumSize(900, 700)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0D1117;
            }
            QWidget {
                color: #E6EDF3;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton {
                background-color: #21262D;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 8px 16px;
                color: #E6EDF3;
            }
            QPushButton:hover {
                background-color: #30363D;
                border-color: #58A6FF;
            }
            QLabel {
                color: #8B949E;
            }
        """)
        
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        
        toolbar = QtWidgets.QHBoxLayout()
        
        load_btn = QtWidgets.QPushButton("📂 Load Model")
        load_btn.clicked.connect(self.load_model)
        toolbar.addWidget(load_btn)
        
        reset_btn = QtWidgets.QPushButton("🔄 Reset View")
        reset_btn.clicked.connect(self.reset_view)
        toolbar.addWidget(reset_btn)
        
        auto_rotate_btn = QtWidgets.QPushButton("🔁 Auto Rotate")
        auto_rotate_btn.clicked.connect(self.toggle_auto_rotate)
        toolbar.addWidget(auto_rotate_btn)
        
        toolbar.addStretch()
        
        help_label = QtWidgets.QLabel("Mouse: LMB=Rotate | RMB=Pan | Scroll=Zoom | R=Reset | A=Auto-rotate")
        toolbar.addWidget(help_label)
        
        layout.addLayout(toolbar)
        
        self.viewer = ModelViewer()
        layout.addWidget(self.viewer, stretch=1)
        
    def load_model(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load 3D Model", "",
            "3D Models (*.obj *.stl *.ply);;All Files (*)"
        )
        if filepath:
            self.viewer.load_model(filepath)
            
    def reset_view(self):
        self.viewer.reset_view()
        
    def toggle_auto_rotate(self):
        self.viewer.auto_rotate = not self.viewer.auto_rotate


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ModelViewerWindow()
    window.show()
    sys.exit(app.exec_())
