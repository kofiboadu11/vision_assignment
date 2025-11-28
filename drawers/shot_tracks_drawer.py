import cv2

class ShotDrawer:
    def __init__(self):
        # How many frames the alert stays on screen after detection
        self.display_frames = 20 
        self.current_display_count = 0

    def draw(self, video_frames, shot_frames):
        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            # Check if a shot happened in this specific frame
            if shot_frames[frame_num]:
                self.current_display_count = self.display_frames

            # If we are currently in the "display window"
            if self.current_display_count > 0:
                text = "SHOT DETECTED!"
                
                # Draw black background rectangle
                cv2.rectangle(frame, (40, 40), (400, 100), (0, 0, 0), cv2.FILLED)
                # Draw text
                cv2.putText(
                    frame, 
                    text, 
                    (50, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, 
                    (0, 255, 0), # Green text
                    3
                )
                
                self.current_display_count -= 1
            
            output_video_frames.append(frame)
            
        return output_video_frames