import cv2
import numpy as np

class PoseDrawer:
    """
    Draws pose estimation skeleton on the shooter.
    Uses YOLO pose keypoints format (17 keypoints for COCO pose).
    """

    # YOLO pose uses COCO 17 keypoints format:
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

    # Define skeleton connections (bone structure)
    SKELETON = [
        (0, 1), (0, 2),  # nose to eyes
        (1, 3), (2, 4),  # eyes to ears
        (0, 5), (0, 6),  # nose to shoulders
        (5, 6),          # shoulders
        (5, 7), (7, 9),  # left arm
        (6, 8), (8, 10), # right arm
        (5, 11), (6, 12), # shoulders to hips
        (11, 12),        # hips
        (11, 13), (13, 15), # left leg
        (12, 14), (14, 16)  # right leg
    ]

    # Color scheme for different body parts
    COLORS = {
        'head': (255, 100, 100),      # Light blue for head
        'torso': (100, 255, 100),     # Light green for torso
        'left_arm': (255, 150, 50),   # Orange for left arm
        'right_arm': (50, 150, 255),  # Blue for right arm
        'left_leg': (255, 255, 100),  # Yellow for left leg
        'right_leg': (100, 255, 255)  # Cyan for right leg
    }

    def __init__(self, keypoint_radius=4, line_thickness=2, min_confidence=0.3):
        """
        Initialize pose drawer.

        Args:
            keypoint_radius: Radius of keypoint circles
            line_thickness: Thickness of skeleton lines
            min_confidence: Minimum confidence threshold to draw a keypoint
        """
        self.keypoint_radius = keypoint_radius
        self.line_thickness = line_thickness
        self.min_confidence = min_confidence

    def get_connection_color(self, idx1, idx2):
        """Get color for a skeleton connection based on body part."""
        # Head connections
        if (idx1, idx2) in [(0, 1), (0, 2), (1, 3), (2, 4)]:
            return self.COLORS['head']
        # Left arm
        elif (idx1, idx2) in [(5, 7), (7, 9)]:
            return self.COLORS['left_arm']
        # Right arm
        elif (idx1, idx2) in [(6, 8), (8, 10)]:
            return self.COLORS['right_arm']
        # Left leg
        elif (idx1, idx2) in [(11, 13), (13, 15)]:
            return self.COLORS['left_leg']
        # Right leg
        elif (idx1, idx2) in [(12, 14), (14, 16)]:
            return self.COLORS['right_leg']
        # Torso (default)
        else:
            return self.COLORS['torso']

    def draw(self, video_frames, pose_tracks, player_tracks):
        """
        Draw pose estimation on video frames for the shooter.

        Args:
            video_frames: List of video frames
            pose_tracks: Pose tracking data from PoseTracker
            player_tracks: Player tracking data (to get shooter bbox)

        Returns:
            Video frames with pose drawn on shooter
        """
        output_frames = video_frames.copy()

        for frame_num, frame in enumerate(output_frames):
            if frame_num >= len(pose_tracks):
                continue

            frame_pose_data = pose_tracks[frame_num]

            # Draw pose for each tracked player (shooter)
            for track_id, pose_data in frame_pose_data.items():
                keypoints = pose_data.get('keypoints', [])
                confidence = pose_data.get('confidence', [])

                if not keypoints or not confidence:
                    continue

                # Convert to numpy arrays for easier handling
                keypoints = np.array(keypoints)
                confidence = np.array(confidence)

                # Draw skeleton connections first (so they appear behind keypoints)
                for idx1, idx2 in self.SKELETON:
                    if (idx1 < len(keypoints) and idx2 < len(keypoints) and
                        confidence[idx1] > self.min_confidence and
                        confidence[idx2] > self.min_confidence):

                        pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                        pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))

                        color = self.get_connection_color(idx1, idx2)
                        cv2.line(frame, pt1, pt2, color, self.line_thickness)

                # Draw keypoints on top
                for i, (kp, conf) in enumerate(zip(keypoints, confidence)):
                    if conf > self.min_confidence:
                        x, y = int(kp[0]), int(kp[1])
                        # Draw keypoint circle
                        cv2.circle(frame, (x, y), self.keypoint_radius, (0, 255, 0), -1)
                        # Draw small border for visibility
                        cv2.circle(frame, (x, y), self.keypoint_radius + 1, (0, 0, 0), 1)

                # Add label "SHOOTER" above the person
                if frame_num < len(player_tracks):
                    player_data = player_tracks[frame_num].get(track_id, {})
                    if "bbox" in player_data:
                        bbox = player_data["bbox"]
                        x1 = int(bbox[0])
                        y1 = int(bbox[1]) - 10

                        # Draw background rectangle for text
                        text = "SHOOTER"
                        (text_width, text_height), baseline = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        cv2.rectangle(
                            frame,
                            (x1, y1 - text_height - 5),
                            (x1 + text_width + 5, y1 + 5),
                            (0, 255, 0),
                            -1
                        )
                        # Draw text
                        cv2.putText(
                            frame,
                            text,
                            (x1 + 2, y1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                            2
                        )

        return output_frames
