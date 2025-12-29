# Cam2Mouse

<div align="center">

![Version](https://img.shields.io/badge/Version-2.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Transform your hand gestures into seamless computer control**

*Experience the future of touchless interaction with enhanced accuracy, two-hand support, and 3D model manipulation*

</div>

---

## ✨ What's New in 2.0

- 🖐️ **Two-Hand Tracking** - Independent gesture recognition for both hands
- 🎯 **Kalman Filtering** - Smooth, accurate cursor movement with reduced jitter
- 🎮 **3D Model Interaction** - Rotate, scale, and translate 3D models with gestures
- 🖼️ **Picture-in-Picture Overlay** - Floating camera feed that stays on top
- ⚡ **Multi-threaded Processing** - Improved performance with dedicated frame processing
- 🎨 **Modern Dark UI** - Beautiful, professional interface design

---

## 🎬 Features

### Cursor Control
| Gesture | Action |
|---------|--------|
| 👆 Index Finger | Move cursor |
| 👌 Pinch (Thumb + Index) | Left click |
| 🤏 Hold Pinch | Click and drag |
| 🤙 Pinky to Wrist | Right click |
| ✌️ Two Fingers Together | Scroll up/down |
| 🖐️ Open Palm | Stop action |

### 3D Model Control (with 3D Mode enabled)
| Gesture | Action |
|---------|--------|
| ✊ Left Fist + Move | Rotate model |
| 🤏 Both Hands Pinch | Scale model |
| 🖐️ Right Open Palm | Pan/Translate |

<div align="center">

### Gesture Visual Guide

| Left Click | Right Click | Drag |
|:----------:|:-----------:|:----:|
| <img src="pictures/left-click-gesture.png" width="150"> | <img src="pictures/right-click-gesture.png" width="150"> | <img src="pictures/ok-gesture.png" width="150"> |

| Scroll Up | Scroll Down |
|:---------:|:-----------:|
| <img src="pictures/scroll-up-gesture.png" width="150"> | <img src="pictures/scroll-down-gesture.png" width="150"> |

</div>

---

## 🚀 Installation

### Requirements

- Python 3.10 or higher
- Webcam
- Windows 10/11 or Linux (X11)

### Windows

```bash
# Clone the repository
git clone https://github.com/aleafarrel-id/alea-aircursor.git
cd alea-aircursor

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Linux

```bash
# Clone the repository
git clone https://github.com/aleafarrel-id/alea-aircursor.git
cd alea-aircursor

# Install system dependencies
sudo apt update && sudo apt install v4l-utils python3-opengl

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r linux-requirements.txt
```

---

## 📖 Usage

### Starting the Application

```bash
python main.py
```

### Quick Start Guide

1. **Select Camera** - Choose your webcam from the dropdown
2. **Start Tracking** - Click "Start Tracking" to begin
3. **Position Hand** - Keep your hand visible to the camera
4. **Control Cursor** - Use your index finger to move the cursor
5. **Perform Gestures** - Use the gestures listed above

### Picture-in-Picture Mode

The floating camera overlay shows your hand position in real-time:
- Drag to reposition
- Click X to hide
- Toggle via "Toggle Overlay" button

### 3D Mode

Enable 3D Mode to interact with 3D models:
1. Check "Enable 3D Mode"
2. Load a model (File → Load 3D Model)
3. Use two-hand gestures to manipulate

Supported formats: `.obj`, `.stl`, `.ply`

---

## ⚙️ Configuration

### Gesture Sensitivity

The application uses adaptive sensitivity based on hand movement speed. For fine-tuning, modify these values in `gesture.py`:

```python
HandTracker(
    click_radius=25,      # Distance threshold for clicks
    hover_radius=35,      # Distance for hover feedback
    scroll_speed=80,      # Scroll sensitivity
    cooldown=0.3,         # Time between repeated actions
)
```

### Kalman Filter Tuning

For cursor smoothness vs responsiveness:

```python
KalmanFilter1D(
    q=0.05,  # Lower = smoother, higher = more responsive
    r=0.3,   # Measurement noise
)
```

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Dual-core 2.0 GHz | Quad-core 3.0 GHz |
| RAM | 4 GB | 8 GB |
| Camera | 720p 30fps | 1080p 60fps |
| GPU | Integrated | Dedicated (for 3D) |

---

## 🐛 Troubleshooting

### Camera Not Detected
- Ensure camera is connected and not in use by another application
- Try selecting a different camera from the dropdown
- On Linux, check permissions: `sudo usermod -a -G video $USER`

### Jittery Cursor Movement
- Ensure good lighting conditions
- Keep hand at a consistent distance from camera
- Increase the Kalman filter smoothing (lower `q` value)

### High CPU Usage
- Reduce camera resolution in settings
- Disable 3D mode if not needed
- Close other resource-intensive applications

### Wayland Compatibility (Linux)
> ⚠️ This application works best with X11. On Wayland, use XWayland:
> ```bash
> QT_QPA_PLATFORM=xcb python main.py
> ```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Hand tracking framework
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework
- [PyOpenGL](http://pyopengl.sourceforge.net/) - 3D rendering
- Original concept by [Alea Farrel](https://github.com/aleafarrel-id)

---

<div align="center">

[Report Bug](https://github.com/aleafarrel-id/alea-aircursor/issues) · [Request Feature](https://github.com/aleafarrel-id/alea-aircursor/issues)

</div>
