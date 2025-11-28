
import cv2

class ShotDrawer:
    def __init__(self):
        # How many frames the alert stays on screen after detection
        self.display_frames = 60  # Stays visible for 60 frames
        self.current_display_count = 0
        # How many frames of trajectory to show before the shot
        # Free throw ball flight is ~1 second, which at 24fps = 24 frames
        self.trajectory_length = 60
        # Store the frozen trajectory points
        self.frozen_trajectory = []
        # Store the current shot result
        self.current_shot_result = None

    def draw(self, video_frames, shot_frames, ball_tracks, shot_results):
        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            # Check if a shot happened in this specific frame
            if shot_frames[frame_num]:
                self.current_display_count = self.display_frames
                # Capture the trajectory at the moment of shot detection
                self.frozen_trajectory = self._get_trajectory_points(frame_num, ball_tracks)
                # Store the shot result (made or missed)
                self.current_shot_result = shot_results[frame_num]

            # If we are currently in the "display window"
            if self.current_display_count > 0:
                # Draw the FROZEN trajectory
                frame = self._draw_frozen_trajectory(frame)
                
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
                
                # Draw background rectangle
                cv2.rectangle(frame, (40, 40), (450, 100), bg_color, cv2.FILLED)
                # Draw text
                cv2.putText(
                    frame, 
                    text, 
                    (50, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, 
                    text_color,
                    3
                )
                
                self.current_display_count -= 1
                
                # Clear frozen trajectory when display count reaches 0
                if self.current_display_count == 0:
                    self.frozen_trajectory = []
                    self.current_shot_result = None
            
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
    
    def _draw_frozen_trajectory(self, frame):
        """
        Draw the frozen trajectory that was captured at shot detection.
        Color changes based on shot result.
        """
        if len(self.frozen_trajectory) > 1:
            # Choose trajectory color based on result
            if self.current_shot_result == "made":
                trajectory_color = (0, 255, 0)  # Green for made shots
            elif self.current_shot_result == "missed":
                trajectory_color = (0, 0, 255)  # Red for missed shots
            else:
                trajectory_color = (0, 255, 255)  # Yellow for unknown
            
            for i in range(1, len(self.frozen_trajectory)):
                # Make the trail fade: older points are thinner
                alpha = i / len(self.frozen_trajectory)
                thickness = int(2 + alpha * 3)  # Thickness from 2 to 5
                
                cv2.line(
                    frame,
                    self.frozen_trajectory[i-1],
                    self.frozen_trajectory[i],
                    trajectory_color,
                    thickness,
                    cv2.LINE_AA
                )
                
                # Optional: Draw small circles at each point for a dotted effect
                cv2.circle(
                    frame,
                    self.frozen_trajectory[i],
                    int(2 + alpha * 2),
                    trajectory_color,
                    -1
                )
        
        return frame