"""
AirCursor 2.0 Configuration
============================
Central configuration file for easy customization of all settings.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class GestureConfig:
    """Gesture detection thresholds and settings"""
    
    # General timing
    cooldown: float = 0.3  # Seconds between repeated actions
    
    # Left click gesture (thumb to index finger)
    click_radius: int = 25  # Distance threshold for click detection
    hover_radius: int = 35  # Distance threshold for hover feedback
    
    # Right click gesture (pinky to wrist)
    right_click_radius: int = 45
    right_hover_radius: int = 55
    
    # Click and hold gesture (OK sign)
    hold_click_radius: int = 35
    hold_hover_radius: int = 45
    
    # Scroll gesture (two fingers)
    scroll_radius: int = 30
    scroll_hover_radius: int = 40
    scroll_speed: int = 80
    scroll_interval: float = 0.05  # Seconds between scroll events
    
    # 3D gesture settings
    rotation_sensitivity: float = 0.5
    scale_sensitivity: float = 0.01
    translation_sensitivity: float = 0.01


@dataclass
class TrackingConfig:
    """Hand tracking settings"""
    
    # MediaPipe settings
    max_hands: int = 2  # Maximum number of hands to track
    model_complexity: int = 1  # 0, 1, or 2 (higher = more accurate but slower)
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.6
    
    # Kalman filter settings
    kalman_process_noise: float = 0.05  # Lower = smoother cursor
    kalman_measurement_noise: float = 0.3  # Higher = more filtering
    
    # Position smoothing
    position_history_size: int = 5  # Number of frames for moving average
    
    # Adaptive sensitivity
    velocity_acceleration_factor: float = 0.1
    max_acceleration: float = 2.0


@dataclass
class CameraConfig:
    """Camera settings"""
    
    # Resolution
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 60
    
    # Buffer settings
    buffer_size: int = 1  # Lower = less latency
    
    # Display
    flip_horizontal: bool = True  # Mirror the camera feed


@dataclass
class OverlayConfig:
    """Picture-in-picture overlay settings"""
    
    # Size
    width: int = 320
    height: int = 240
    
    # Position (from screen edge)
    margin_right: int = 20
    margin_bottom: int = 80
    
    # Appearance
    background_opacity: float = 0.9
    border_radius: int = 12
    border_color: Tuple[int, int, int] = (100, 200, 255)


@dataclass
class ModelViewerConfig:
    """3D model viewer settings"""
    
    # Initial view
    initial_rotation_x: float = 30.0
    initial_rotation_y: float = 45.0
    initial_zoom: float = 3.0
    
    # Limits
    min_zoom: float = 0.5
    max_zoom: float = 10.0
    min_scale: float = 0.1
    max_scale: float = 5.0
    
    # Animation
    auto_rotate_speed: float = 0.5  # Degrees per frame
    
    # Appearance
    background_color: Tuple[float, float, float] = (0.1, 0.1, 0.15)
    model_color: Tuple[float, float, float] = (0.6, 0.6, 0.8)
    grid_color: Tuple[float, float, float] = (0.3, 0.3, 0.3)


@dataclass
class UIConfig:
    """User interface settings"""
    
    # Window
    min_width: int = 900
    min_height: int = 750
    
    # Colors (dark theme)
    background_primary: str = "#0D1117"
    background_secondary: str = "#161B22"
    background_tertiary: str = "#21262D"
    
    border_color: str = "#30363D"
    accent_color: str = "#58A6FF"
    success_color: str = "#3FB950"
    error_color: str = "#F85149"
    
    text_primary: str = "#E6EDF3"
    text_secondary: str = "#8B949E"


# Default configurations
DEFAULT_GESTURE_CONFIG = GestureConfig()
DEFAULT_TRACKING_CONFIG = TrackingConfig()
DEFAULT_CAMERA_CONFIG = CameraConfig()
DEFAULT_OVERLAY_CONFIG = OverlayConfig()
DEFAULT_MODEL_VIEWER_CONFIG = ModelViewerConfig()
DEFAULT_UI_CONFIG = UIConfig()


def load_config(filepath: str = None) -> dict:
    """
    Load configuration from a JSON file.
    Falls back to defaults if file doesn't exist.
    """
    import json
    import os
    
    config = {
        'gesture': DEFAULT_GESTURE_CONFIG,
        'tracking': DEFAULT_TRACKING_CONFIG,
        'camera': DEFAULT_CAMERA_CONFIG,
        'overlay': DEFAULT_OVERLAY_CONFIG,
        'model_viewer': DEFAULT_MODEL_VIEWER_CONFIG,
        'ui': DEFAULT_UI_CONFIG,
    }
    
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                user_config = json.load(f)
                
            # Update configs with user values
            for section, values in user_config.items():
                if section in config:
                    for key, value in values.items():
                        if hasattr(config[section], key):
                            setattr(config[section], key, value)
        except Exception as e:
            print(f"Error loading config: {e}")
    
    return config


def save_config(config: dict, filepath: str):
    """Save configuration to a JSON file."""
    import json
    from dataclasses import asdict
    
    try:
        output = {}
        for section, cfg in config.items():
            output[section] = asdict(cfg)
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")
