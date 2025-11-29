
import cv2
import numpy as np

class ShotDrawer:
    def __init__(self):
        # How many frames the alert stays on screen after detection
        self.display_frames = 60  # Stays visible for 60 frames
        self.current_display_count = 0
        # How many frames of trajectory to show before the shot
        self.trajectory_length = 90  # Extended to capture full shot arc
        # Store the frozen trajectory points
        self.frozen_trajectory = []
        # Store the current shot result
        self.current_shot_result = None
        # Store release point and peak point
        self.release_point = None
        self.peak_point = None
        # Store predicted trajectory
        self.predicted_trajectory = []

    def draw(self, video_frames, shot_frames, ball_tracks, shot_results):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # Check if a shot happened in this specific frame
            if shot_frames[frame_num]:
                self.current_display_count = self.display_frames
                # Capture the trajectory at the moment of shot detection
                self.frozen_trajectory = self._get_trajectory_points(frame_num, ball_tracks)
                # Analyze trajectory to find key points
                self.release_point, self.peak_point = self._analyze_trajectory(self.frozen_trajectory)
                # Generate predicted trajectory arc
                self.predicted_trajectory = self._predict_trajectory_arc(self.frozen_trajectory)
                # Store the shot result (made or missed)
                self.current_shot_result = shot_results[frame_num]

            # If we are currently in the "display window"
            if self.current_display_count > 0:
                # Draw the enhanced trajectory
                frame = self._draw_enhanced_trajectory(frame)

                # Determine text and color based on shot result
                if self.current_shot_result == "made":
                    text = "SHOT MADE!"
                    text_color = (0, 255, 0)  # Green
                    bg_color = (0, 100, 0)    # Dark green background
                elif self.current_shot_result == "missed":
                    text = "SHOT MISSED!"
                    text_color = (0, 0, 255)  # Red
                    bg_color = (0, 0, 100)    # Dark red background
                else:
                    text = "SHOT DETECTED!"
                    text_color = (0, 255, 0)  # Green
                    bg_color = (0, 0, 0)      # Black background

                # Draw background rectangle with transparency effect
                overlay = frame.copy()
                cv2.rectangle(overlay, (40, 40), (450, 100), bg_color, cv2.FILLED)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                # Draw text with outline for better visibility
                self._draw_text_with_outline(
                    frame, text, (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    text_color, (0, 0, 0), 3, 5
                )

                self.current_display_count -= 1

                # Clear frozen trajectory when display count reaches 0
                if self.current_display_count == 0:
                    self.frozen_trajectory = []
                    self.current_shot_result = None
                    self.release_point = None
                    self.peak_point = None
                    self.predicted_trajectory = []

            output_video_frames.append(frame)

        return output_video_frames
    
    def _get_trajectory_points(self, current_frame_num, ball_tracks):
        """
        Get the ball's trajectory points for the last N frames.
        This is called once when a shot is detected.
        """
        trajectory_points = []
        
        # Go back trajectory_length frames from current position
        start_frame = max(0, current_frame_num - self.trajectory_length)
        
        for frame_idx in range(start_frame, current_frame_num + 1):
            if frame_idx < len(ball_tracks):
                ball_data = ball_tracks[frame_idx].get(1, {})
                if "bbox" in ball_data:
                    bbox = ball_data["bbox"]
                    # Get center of ball
                    center_x = int((bbox[0] + bbox[2]) / 2)
                    center_y = int((bbox[1] + bbox[3]) / 2)
                    trajectory_points.append((center_x, center_y))
        
        return trajectory_points
    
    def _analyze_trajectory(self, trajectory):
        """
        Analyze trajectory to find release point and peak point.
        """
        if len(trajectory) < 3:
            return None, None

        # Release point is the first point
        release_point = trajectory[0]

        # Find peak point (highest y-value, which is lowest in image coordinates)
        peak_idx = 0
        min_y = trajectory[0][1]
        for i, point in enumerate(trajectory):
            if point[1] < min_y:
                min_y = point[1]
                peak_idx = i

        peak_point = trajectory[peak_idx]
        return release_point, peak_point

    def _predict_trajectory_arc(self, trajectory):
        """
        Generate a smooth predicted trajectory arc using polynomial fitting.
        """
        if len(trajectory) < 5:
            return []

        # Extract x and y coordinates
        x_coords = np.array([p[0] for p in trajectory])
        y_coords = np.array([p[1] for p in trajectory])

        # Fit a 2nd degree polynomial (parabola)
        try:
            coeffs = np.polyfit(x_coords, y_coords, 2)
            poly = np.poly1d(coeffs)

            # Generate smooth arc points
            x_min, x_max = x_coords.min(), x_coords.max()
            x_smooth = np.linspace(x_min, x_max, 100)
            y_smooth = poly(x_smooth)

            predicted_points = [(int(x), int(y)) for x, y in zip(x_smooth, y_smooth)]
            return predicted_points
        except:
            return []

    def _draw_text_with_outline(self, frame, text, position, font, scale,
                                 text_color, outline_color, thickness, outline_thickness):
        """
        Draw text with an outline for better visibility.
        """
        # Draw outline
        cv2.putText(frame, text, position, font, scale, outline_color, outline_thickness)
        # Draw text
        cv2.putText(frame, text, position, font, scale, text_color, thickness)

    def _draw_enhanced_trajectory(self, frame):
        """
        Draw enhanced trajectory with gradient colors, key points, and smooth arc.
        """
        if len(self.frozen_trajectory) < 2:
            return frame

        # Choose base color based on shot result
        if self.current_shot_result == "made":
            base_color = np.array([0, 255, 0])  # Green
            glow_color = (100, 255, 100)
        elif self.current_shot_result == "missed":
            base_color = np.array([0, 0, 255])  # Red
            glow_color = (100, 100, 255)
        else:
            base_color = np.array([0, 255, 255])  # Yellow
            glow_color = (100, 255, 255)

        # Draw predicted arc first (semi-transparent)
        if len(self.predicted_trajectory) > 1:
            overlay = frame.copy()
            for i in range(1, len(self.predicted_trajectory)):
                alpha = i / len(self.predicted_trajectory)
                cv2.line(
                    overlay,
                    self.predicted_trajectory[i-1],
                    self.predicted_trajectory[i],
                    tuple(map(int, base_color * 0.5)),
                    1,
                    cv2.LINE_AA
                )
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Draw actual trajectory with gradient and glow effect
        for i in range(1, len(self.frozen_trajectory)):
            # Calculate gradient: start dim, end bright
            alpha = i / len(self.frozen_trajectory)

            # Interpolate color for gradient effect
            current_color = tuple(map(int, base_color * (0.4 + alpha * 0.6)))

            # Variable thickness: thicker as we progress
            thickness = int(2 + alpha * 4)

            pt1 = self.frozen_trajectory[i-1]
            pt2 = self.frozen_trajectory[i]

            # Draw glow effect (thicker, lighter line underneath)
            cv2.line(frame, pt1, pt2, glow_color, thickness + 4, cv2.LINE_AA)

            # Draw main trajectory line
            cv2.line(frame, pt1, pt2, current_color, thickness, cv2.LINE_AA)

            # Draw circles at each point for emphasis
            circle_radius = int(2 + alpha * 3)
            cv2.circle(frame, pt2, circle_radius + 2, glow_color, -1)
            cv2.circle(frame, pt2, circle_radius, current_color, -1)

        # Highlight release point
        if self.release_point:
            cv2.circle(frame, self.release_point, 8, (255, 255, 255), 2)
            cv2.circle(frame, self.release_point, 5, tuple(map(int, base_color)), -1)
            # Label
            cv2.putText(
                frame, "RELEASE",
                (self.release_point[0] - 30, self.release_point[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )

        # Highlight peak point
        if self.peak_point and self.peak_point != self.release_point:
            cv2.circle(frame, self.peak_point, 8, (255, 255, 255), 2)
            cv2.circle(frame, self.peak_point, 5, (255, 200, 0), -1)
            # Label
            cv2.putText(
                frame, "PEAK",
                (self.peak_point[0] - 20, self.peak_point[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )

        return frame