"""
Enhanced Hand Tracker with Kalman Filtering, Two-Hand Support, and 3D Gestures
===============================================================================
Features:
- Kalman filtering for smooth, accurate cursor movement
- Two-hand independent tracking
- 3D model manipulation gestures
- Optimized performance with reduced latency
- Platform-specific optimizations for Windows and Linux
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import platform
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

pyautogui.FAILSAFE = False


@dataclass
class KalmanFilter1D:
    """1D Kalman filter for smooth position tracking"""
    q: float = 0.1  # Process noise
    r: float = 0.5  # Measurement noise
    x: float = 0.0  # State estimate
    p: float = 1.0  # Estimate uncertainty
    
    def update(self, measurement: float) -> float:
        # Prediction
        self.p += self.q
        
        # Update
        k = self.p / (self.p + self.r)  # Kalman gain
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)
        
        return self.x


@dataclass
class HandState:
    """State for a single hand"""
    detected: bool = False
    gesture: str = "None"
    landmarks: List = field(default_factory=list)
    
    # Cursor position with Kalman filtering
    kalman_x: KalmanFilter1D = field(default_factory=lambda: KalmanFilter1D(q=0.05, r=0.3))
    kalman_y: KalmanFilter1D = field(default_factory=lambda: KalmanFilter1D(q=0.05, r=0.3))
    
    # Previous positions for velocity calculation
    prev_x: float = 0.0
    prev_y: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    
    # Gesture timing
    last_click_time: float = 0.0
    last_action_time: float = 0.0
    is_holding: bool = False
    is_scrolling: bool = False
    scroll_direction: Optional[str] = None
    
    # 3D interaction state
    rotation_origin: Optional[Tuple[float, float]] = None
    scale_origin_distance: float = 0.0
    translation_origin: Optional[Tuple[float, float]] = None
    
    def reset(self):
        self.detected = False
        self.gesture = "None"
        self.landmarks = []


class HandTracker:
    def __init__(
        self,
        cooldown: float = 0.3,
        complexity: int = 1,
        click_radius: int = 25,
        hover_radius: int = 35,
        right_click_radius: int = 45,
        right_hover_radius: int = 55,
        hold_click_radius: int = 35,
        hold_hover_radius: int = 45,
        scroll_radius: int = 30,
        scroll_hover_radius: int = 40,
        scroll_speed: int = 80,
        enable_two_hands: bool = True,
        enable_3d_mode: bool = False
    ):
        """Initialize the enhanced Hand Tracker"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2 if enable_two_hands else 1,
            model_complexity=complexity,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        
        # Configuration
        self.cooldown = cooldown
        self.click_radius = click_radius
        self.hover_radius = hover_radius
        self.right_click_radius = right_click_radius
        self.right_hover_radius = right_hover_radius
        self.hold_click_radius = hold_click_radius
        self.hold_hover_radius = hold_hover_radius
        self.scroll_radius = scroll_radius
        self.scroll_hover_radius = scroll_hover_radius
        self.scroll_speed = scroll_speed
        self.enable_two_hands = enable_two_hands
        self.enable_3d_mode = enable_3d_mode
        
        # Hand states
        self.left_hand = HandState()
        self.right_hand = HandState()
        
        # Landmark indices for drawing
        self.tracked_landmarks = [0, 4, 5, 6, 7, 8, 9, 12, 16, 20]
        self.landmark_colors: Dict[str, Dict[int, Tuple[int, int, int]]] = {
            'Left': {i: (0, 255, 0) for i in range(21)},
            'Right': {i: (255, 165, 0) for i in range(21)}
        }
        
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Scroll thread
        self.scroll_thread = None
        self.scroll_active = False
        self.scroll_lock = threading.Lock()
        self.scroll_interval = 0.05
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Position smoothing
        self.position_history_x = deque(maxlen=5)
        self.position_history_y = deque(maxlen=5)
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict, Dict]:
        """Process frame and return processed image with hand data"""
        current_time = time.time()
        
        # Calculate FPS
        self.frame_times.append(current_time - self.last_frame_time)
        self.last_frame_time = current_time
        fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0
        
        # Mirror frame
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        try:
            results = self.hands.process(rgb_frame)
        except Exception as e:
            print(f"MediaPipe processing error: {e}")
            return frame, {'detected': False}, {'detected': False}
        
        rgb_frame.flags.writeable = True
        
        # Reset hand states
        self.left_hand.reset()
        self.right_hand.reset()
        
        left_hand_data = {'detected': False, 'fps': fps}
        right_hand_data = {'detected': False, 'fps': fps}
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # Determine hand type
                hand_label = handedness.classification[0].label
                hand_state = self.left_hand if hand_label == 'Left' else self.right_hand
                colors = self.landmark_colors[hand_label]
                
                # Reset colors
                for i in range(21):
                    colors[i] = (0, 255, 0) if hand_label == 'Left' else (255, 165, 0)
                
                # Extract landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((cx, cy, lm.z))
                
                hand_state.detected = True
                hand_state.landmarks = landmarks
                
                # Process gestures based on hand
                if hand_label == 'Right':
                    # Right hand controls cursor
                    right_hand_data = self.process_right_hand(
                        hand_state, landmarks, colors, current_time, w, h
                    )
                    right_hand_data['fps'] = fps
                else:
                    # Left hand for secondary actions and 3D control
                    left_hand_data = self.process_left_hand(
                        hand_state, landmarks, colors, current_time, w, h
                    )
                    left_hand_data['fps'] = fps
                
                # Draw landmarks
                self.draw_hand_landmarks(frame, landmarks, colors, hand_label)
        else:
            # No hands detected - release any active holds
            if self.right_hand.is_holding:
                pyautogui.mouseUp()
                self.right_hand.is_holding = False
            self.stop_scrolling()
        
        # Handle 3D interactions with both hands
        if self.enable_3d_mode and self.left_hand.detected and self.right_hand.detected:
            self.process_3d_gestures(left_hand_data, right_hand_data)
        
        # Draw status overlay
        self.draw_status_overlay(frame, fps, left_hand_data, right_hand_data)
        
        return frame, left_hand_data, right_hand_data
    
    def process_right_hand(
        self,
        hand_state: HandState,
        landmarks: List,
        colors: Dict[int, Tuple[int, int, int]],
        current_time: float,
        frame_w: int,
        frame_h: int
    ) -> Dict:
        """Process right hand for cursor control and clicking"""
        data = {'detected': True, 'gesture': 'Pointer'}
        
        if len(landmarks) < 21:
            return data
        
        # Get index finger tip for cursor movement
        index_tip = landmarks[8]
        
        # Apply Kalman filtering for smooth cursor movement
        filtered_x = hand_state.kalman_x.update(index_tip[0])
        filtered_y = hand_state.kalman_y.update(index_tip[1])
        
        # Add to position history for additional smoothing
        self.position_history_x.append(filtered_x)
        self.position_history_y.append(filtered_y)
        
        # Weighted moving average
        if len(self.position_history_x) > 1:
            weights = np.linspace(0.5, 1.0, len(self.position_history_x))
            weights /= weights.sum()
            smooth_x = np.average(list(self.position_history_x), weights=weights)
            smooth_y = np.average(list(self.position_history_y), weights=weights)
        else:
            smooth_x, smooth_y = filtered_x, filtered_y
        
        # Calculate velocity for adaptive sensitivity
        hand_state.velocity_x = smooth_x - hand_state.prev_x
        hand_state.velocity_y = smooth_y - hand_state.prev_y
        hand_state.prev_x = smooth_x
        hand_state.prev_y = smooth_y
        
        # Map to screen coordinates with adaptive acceleration
        velocity_magnitude = np.sqrt(hand_state.velocity_x**2 + hand_state.velocity_y**2)
        acceleration = 1.0 + min(velocity_magnitude * 0.1, 2.0)
        
        cursor_x = np.interp(smooth_x, [0, frame_w], [0, self.screen_w])
        cursor_y = np.interp(smooth_y, [0, frame_h], [0, self.screen_h])
        
        # Move cursor
        pyautogui.moveTo(cursor_x, cursor_y, _pause=False)
        
        # Gesture detection
        thumb_tip = landmarks[4]
        
        # Scroll detection (two fingers)
        scroll_detected = self.detect_scroll(landmarks, colors, current_time, hand_state)
        if scroll_detected:
            data['gesture'] = 'Scrolling'
            hand_state.gesture = 'Scrolling'
            return data
        
        # Click and hold detection
        hold_detected = self.detect_click_and_hold(
            landmarks, colors, current_time, hand_state, thumb_tip, index_tip
        )
        if hold_detected:
            data['gesture'] = 'Holding'
            hand_state.gesture = 'Holding'
            return data
        
        # Left click detection
        if not hand_state.is_holding:
            click_detected = self.detect_left_click(
                landmarks, colors, current_time, hand_state, thumb_tip
            )
            if click_detected:
                data['gesture'] = 'Click'
                hand_state.gesture = 'Click'
                return data
        
        # Right click detection (pinky to wrist)
        right_click = self.detect_right_click(landmarks, colors, current_time, hand_state)
        if right_click:
            data['gesture'] = 'Right Click'
            hand_state.gesture = 'Right Click'
            return data
        
        hand_state.gesture = 'Pointer'
        return data
    
    def process_left_hand(
        self,
        hand_state: HandState,
        landmarks: List,
        colors: Dict[int, Tuple[int, int, int]],
        current_time: float,
        frame_w: int,
        frame_h: int
    ) -> Dict:
        """Process left hand for secondary actions and 3D control"""
        data = {'detected': True, 'gesture': 'Open'}
        
        if len(landmarks) < 21:
            return data
        
        # Detect open palm (all fingers extended)
        if self.is_open_palm(landmarks):
            data['gesture'] = 'Open Palm'
            hand_state.gesture = 'Open Palm'
            
            # Store position for 3D translation
            if self.enable_3d_mode:
                palm_center = self.get_palm_center(landmarks)
                if hand_state.translation_origin is None:
                    hand_state.translation_origin = palm_center
                else:
                    delta_x = palm_center[0] - hand_state.translation_origin[0]
                    delta_y = palm_center[1] - hand_state.translation_origin[1]
                    data['trans_x'] = delta_x
                    data['trans_y'] = delta_y
                    data['gesture'] = 'translating'
                    hand_state.translation_origin = palm_center
        else:
            hand_state.translation_origin = None
        
        # Detect fist for rotation (3D mode)
        if self.enable_3d_mode and self.is_fist(landmarks):
            wrist = landmarks[0]
            if hand_state.rotation_origin is None:
                hand_state.rotation_origin = (wrist[0], wrist[1])
            else:
                delta_x = wrist[0] - hand_state.rotation_origin[0]
                delta_y = wrist[1] - hand_state.rotation_origin[1]
                data['delta_x'] = delta_x
                data['delta_y'] = delta_y
                data['gesture'] = 'rotating'
                hand_state.rotation_origin = (wrist[0], wrist[1])
            
            # Color feedback
            for i in range(21):
                colors[i] = (255, 0, 255)
        else:
            hand_state.rotation_origin = None
        
        hand_state.gesture = data['gesture']
        return data
    
    def process_3d_gestures(self, left_data: Dict, right_data: Dict):
        """Process two-hand gestures for 3D manipulation"""
        if not (self.left_hand.detected and self.right_hand.detected):
            return
        
        left_landmarks = self.left_hand.landmarks
        right_landmarks = self.right_hand.landmarks
        
        if len(left_landmarks) < 21 or len(right_landmarks) < 21:
            return
        
        # Two-hand scaling: measure distance between index fingers
        left_index = left_landmarks[8]
        right_index = right_landmarks[8]
        
        distance = np.sqrt(
            (left_index[0] - right_index[0])**2 +
            (left_index[1] - right_index[1])**2
        )
        
        # Detect pinch on both hands for scaling
        if self.is_pinch(left_landmarks) and self.is_pinch(right_landmarks):
            if self.left_hand.scale_origin_distance == 0:
                self.left_hand.scale_origin_distance = distance
            else:
                scale_factor = distance / self.left_hand.scale_origin_distance
                left_data['scale_factor'] = scale_factor
                left_data['gesture'] = 'scaling'
                right_data['gesture'] = 'scaling'
                self.left_hand.scale_origin_distance = distance
        else:
            self.left_hand.scale_origin_distance = 0
    
    def detect_scroll(
        self,
        landmarks: List,
        colors: Dict[int, Tuple[int, int, int]],
        current_time: float,
        hand_state: HandState
    ) -> bool:
        """Detect scroll gesture with index and middle fingers"""
        if len(landmarks) < 13:
            self.stop_scrolling()
            return False
        
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        middle_base = landmarks[9]
        
        # Manhattan distance for faster calculation
        distance = abs(index_tip[0] - middle_tip[0]) + abs(index_tip[1] - middle_tip[1])
        
        # Direction based on finger position relative to base
        mid_y = (index_tip[1] + middle_tip[1]) // 2
        vertical_diff = mid_y - middle_base[1]
        direction = 'up' if vertical_diff < -20 else 'down' if vertical_diff > 20 else None
        
        # Hover zone
        if distance < self.scroll_hover_radius:
            colors[8] = (0, 255, 255)
            colors[12] = (0, 255, 255)
            colors[9] = (0, 255, 255)
            
            # Active zone
            if distance < self.scroll_radius and direction:
                colors[8] = (255, 0, 255)
                colors[12] = (255, 0, 255)
                colors[9] = (255, 0, 255)
                
                if not hand_state.is_scrolling:
                    self.start_scrolling(direction)
                    hand_state.is_scrolling = True
                    hand_state.scroll_direction = direction
                elif hand_state.scroll_direction != direction:
                    hand_state.scroll_direction = direction
                    self.scroll_direction = direction
                
                return True
            else:
                if hand_state.is_scrolling:
                    self.stop_scrolling()
                    hand_state.is_scrolling = False
        else:
            if hand_state.is_scrolling:
                self.stop_scrolling()
                hand_state.is_scrolling = False
        
        return False
    
    def detect_click_and_hold(
        self,
        landmarks: List,
        colors: Dict[int, Tuple[int, int, int]],
        current_time: float,
        hand_state: HandState,
        thumb_tip: Tuple,
        index_tip: Tuple
    ) -> bool:
        """Detect click and hold gesture (OK sign)"""
        distance = np.sqrt(
            (thumb_tip[0] - index_tip[0])**2 +
            (thumb_tip[1] - index_tip[1])**2
        )
        
        # Check if middle, ring, pinky are extended (OK gesture)
        middle_extended = landmarks[12][1] < landmarks[10][1]
        ring_extended = landmarks[16][1] < landmarks[14][1]
        pinky_extended = landmarks[20][1] < landmarks[18][1]
        
        if distance < self.hold_hover_radius and (middle_extended or ring_extended):
            colors[4] = (0, 255, 255)
            colors[8] = (0, 255, 255)
            
            if distance < self.hold_click_radius:
                colors[4] = (0, 0, 255)
                colors[8] = (0, 0, 255)
                
                if not hand_state.is_holding:
                    hand_state.is_holding = True
                    pyautogui.mouseDown()
                return True
            else:
                if hand_state.is_holding:
                    hand_state.is_holding = False
                    pyautogui.mouseUp()
        else:
            if hand_state.is_holding:
                hand_state.is_holding = False
                pyautogui.mouseUp()
        
        return False
    
    def detect_left_click(
        self,
        landmarks: List,
        colors: Dict[int, Tuple[int, int, int]],
        current_time: float,
        hand_state: HandState,
        thumb_tip: Tuple
    ) -> bool:
        """Detect left click (thumb to index finger segments)"""
        if current_time - hand_state.last_click_time < self.cooldown:
            return False
        
        contact = False
        for idx in [5, 6, 7]:
            finger_point = landmarks[idx]
            distance = np.sqrt(
                (thumb_tip[0] - finger_point[0])**2 +
                (thumb_tip[1] - finger_point[1])**2
            )
            
            if distance < self.hover_radius:
                colors[4] = (0, 255, 255)
                colors[idx] = (0, 255, 255)
                
                if distance < self.click_radius:
                    colors[4] = (255, 0, 0)
                    colors[idx] = (255, 0, 0)
                    contact = True
        
        if contact:
            pyautogui.click()
            hand_state.last_click_time = current_time
            return True
        
        return False
    
    def detect_right_click(
        self,
        landmarks: List,
        colors: Dict[int, Tuple[int, int, int]],
        current_time: float,
        hand_state: HandState
    ) -> bool:
        """Detect right click (pinky to wrist)"""
        if current_time - hand_state.last_action_time < self.cooldown:
            return False
        
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        
        distance = np.sqrt(
            (pinky_tip[0] - wrist[0])**2 +
            (pinky_tip[1] - wrist[1])**2
        )
        
        if distance < self.right_hover_radius:
            colors[20] = (0, 255, 255)
            colors[0] = (0, 255, 255)
            
            if distance < self.right_click_radius:
                colors[20] = (255, 0, 0)
                colors[0] = (255, 0, 0)
                pyautogui.rightClick()
                hand_state.last_action_time = current_time
                return True
        
        return False
    
    def is_open_palm(self, landmarks: List) -> bool:
        """Check if hand is in open palm position"""
        if len(landmarks) < 21:
            return False
        
        # All fingertips should be above their respective PIP joints
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip][1] > landmarks[pip][1]:  # tip below pip
                return False
        
        return True
    
    def is_fist(self, landmarks: List) -> bool:
        """Check if hand is in fist position"""
        if len(landmarks) < 21:
            return False
        
        # All fingertips should be below their respective MCP joints
        finger_tips = [8, 12, 16, 20]
        finger_mcps = [5, 9, 13, 17]
        
        for tip, mcp in zip(finger_tips, finger_mcps):
            if landmarks[tip][1] < landmarks[mcp][1]:  # tip above mcp
                return False
        
        return True
    
    def is_pinch(self, landmarks: List) -> bool:
        """Check if hand is in pinch position"""
        if len(landmarks) < 9:
            return False
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        distance = np.sqrt(
            (thumb_tip[0] - index_tip[0])**2 +
            (thumb_tip[1] - index_tip[1])**2
        )
        
        return distance < self.hold_click_radius
    
    def get_palm_center(self, landmarks: List) -> Tuple[float, float]:
        """Calculate palm center from landmarks"""
        palm_indices = [0, 1, 5, 9, 13, 17]
        x = sum(landmarks[i][0] for i in palm_indices) / len(palm_indices)
        y = sum(landmarks[i][1] for i in palm_indices) / len(palm_indices)
        return (x, y)
    
    def start_scrolling(self, direction: str):
        """Start continuous scrolling"""
        self.scroll_direction = direction
        
        with self.scroll_lock:
            if not self.scroll_active:
                self.scroll_active = True
                self.scroll_thread = threading.Thread(target=self._scroll_worker, daemon=True)
                self.scroll_thread.start()
    
    def stop_scrolling(self):
        """Stop scrolling"""
        with self.scroll_lock:
            self.scroll_active = False
    
    def _scroll_worker(self):
        """Background scroll worker"""
        while True:
            with self.scroll_lock:
                if not self.scroll_active:
                    break
            self._perform_scroll()
            time.sleep(self.scroll_interval)
    
    def _perform_scroll(self):
        """Platform-specific scroll"""
        os_name = platform.system()
        
        if os_name == 'Windows':
            try:
                import ctypes
                scroll_amount = 60 if self.scroll_direction == 'up' else -60
                ctypes.windll.user32.mouse_event(0x0800, 0, 0, scroll_amount, 0)
            except Exception:
                self._default_scroll()
        elif os_name == 'Linux':
            try:
                from Xlib import display, X
                from Xlib.ext.xtest import fake_input
                
                d = display.Display()
                button = 4 if self.scroll_direction == 'up' else 5
                fake_input(d, X.ButtonPress, button)
                d.sync()
                fake_input(d, X.ButtonRelease, button)
                d.sync()
            except Exception:
                self._default_scroll()
        else:
            self._default_scroll()
    
    def _default_scroll(self):
        """Fallback scroll using pyautogui"""
        scroll_amount = self.scroll_speed if self.scroll_direction == 'up' else -self.scroll_speed
        pyautogui.scroll(scroll_amount // 3)
    
    def draw_hand_landmarks(
        self,
        frame: np.ndarray,
        landmarks: List,
        colors: Dict[int, Tuple[int, int, int]],
        hand_label: str
    ):
        """Draw hand landmarks with connections"""
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        base_color = (0, 200, 0) if hand_label == 'Left' else (200, 130, 0)
        
        for start, end in connections:
            if start < len(landmarks) and end < len(landmarks):
                pt1 = (landmarks[start][0], landmarks[start][1])
                pt2 = (landmarks[end][0], landmarks[end][1])
                cv2.line(frame, pt1, pt2, base_color, 2)
        
        # Draw landmark points
        for idx in self.tracked_landmarks:
            if idx < len(landmarks):
                x, y = landmarks[idx][0], landmarks[idx][1]
                color = colors.get(idx, base_color)
                cv2.circle(frame, (x, y), 8, color, -1)
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 1)
    
    def draw_status_overlay(
        self,
        frame: np.ndarray,
        fps: float,
        left_data: Dict,
        right_data: Dict
    ):
        """Draw status overlay on frame"""
        h, w, _ = frame.shape
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # FPS
        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        
        # Hand status
        left_status = left_data.get('gesture', 'Not detected') if left_data.get('detected') else 'Not detected'
        right_status = right_data.get('gesture', 'Not detected') if right_data.get('detected') else 'Not detected'
        
        cv2.putText(
            frame, f"Left: {left_status}",
            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        cv2.putText(
            frame, f"Right: {right_status}",
            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1
        )
        
        # Mode indicator
        mode = "3D Mode" if self.enable_3d_mode else "2D Mode"
        cv2.putText(
            frame, mode,
            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    
    def release(self):
        """Release resources"""
        with self.scroll_lock:
            self.scroll_active = False
        
        if self.scroll_thread and self.scroll_thread.is_alive():
            self.scroll_thread.join(timeout=0.5)
        
        self.hands.close()
