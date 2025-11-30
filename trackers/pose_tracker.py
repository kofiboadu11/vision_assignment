from ultralytics import YOLO
import numpy as np

class PoseTracker:
    def __init__(self, model_path='yolo11n-pose.pt'):
        """
        Initialize pose tracker with a YOLO pose model.
        Uses YOLO11 pose model by default for human pose estimation.
        """
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        """Run pose detection on all frames in batches."""
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_detections = self.model.predict(batch_frames, conf=0.3)
            detections += batch_detections
        return detections

    def get_shooter_id(self, player_tracks, ball_tracks, frame_num):
        """
        Identify which player is the shooter by finding the player closest to the ball.
        Returns the player track_id who is shooting.
        """
        if frame_num >= len(ball_tracks) or frame_num >= len(player_tracks):
            return None

        ball_data = ball_tracks[frame_num].get(1, {})
        if "bbox" not in ball_data:
            return None

        ball_bbox = ball_data["bbox"]
        ball_center = np.array([
            (ball_bbox[0] + ball_bbox[2]) / 2,
            (ball_bbox[1] + ball_bbox[3]) / 2
        ])

        # Find closest player to ball
        min_distance = float('inf')
        shooter_id = None

        for track_id, player_data in player_tracks[frame_num].items():
            if "bbox" not in player_data:
                continue

            player_bbox = player_data["bbox"]
            player_center = np.array([
                (player_bbox[0] + player_bbox[2]) / 2,
                (player_bbox[1] + player_bbox[3]) / 2
            ])

            distance = np.linalg.norm(ball_center - player_center)
            if distance < min_distance:
                min_distance = distance
                shooter_id = track_id

        return shooter_id

    def detect_shot_sequences(self, ball_tracks):
        """
        Detect shot sequences by analyzing ball movement.
        Returns list of (start_frame, end_frame, is_active) tuples.
        """
        shot_sequences = []
        in_shot = False
        shot_start = None
        shot_window = 60  # Maintain shooter ID for 60 frames (2 seconds at 30fps)

        for frame_num in range(len(ball_tracks) - 1):
            ball_data = ball_tracks[frame_num].get(1, {})
            next_ball_data = ball_tracks[frame_num + 1].get(1, {})

            if "bbox" not in ball_data or "bbox" not in next_ball_data:
                continue

            # Calculate vertical velocity (negative = upward movement)
            ball_y = (ball_data["bbox"][1] + ball_data["bbox"][3]) / 2
            next_ball_y = (next_ball_data["bbox"][1] + next_ball_data["bbox"][3]) / 2
            vertical_velocity = next_ball_y - ball_y

            # Detect start of shot (rapid upward movement)
            if not in_shot and vertical_velocity < -5:  # Ball moving up significantly
                in_shot = True
                shot_start = frame_num

            # End shot sequence after window expires
            if in_shot and frame_num >= shot_start + shot_window:
                shot_sequences.append((shot_start, frame_num))
                in_shot = False
                shot_start = None

        # Close any open shot sequence
        if in_shot and shot_start is not None:
            shot_sequences.append((shot_start, len(ball_tracks) - 1))

        return shot_sequences

    def get_pose_tracks(self, frames, player_tracks, ball_tracks):
        """
        Get pose keypoints for the shooter in each frame.
        Identifies shooter at the start of each shot and maintains that ID throughout.
        Returns a list of dicts with pose data per frame.
        Format: [{track_id: {'keypoints': [...], 'confidence': [...]}}, ...]
        """
        detections = self.detect_frames(frames)
        pose_tracks = []

        # Detect shot sequences
        shot_sequences = self.detect_shot_sequences(ball_tracks)

        # Create a mapping of frame -> shooter_id
        frame_to_shooter = {}
        for start_frame, end_frame in shot_sequences:
            # Identify shooter at the start of the shot
            shooter_id = self.get_shooter_id(player_tracks, ball_tracks, start_frame)
            if shooter_id is not None:
                # Apply this shooter_id to all frames in the shot sequence
                for frame in range(start_frame, end_frame + 1):
                    frame_to_shooter[frame] = shooter_id

        # Generate pose tracks using the locked shooter IDs
        for frame_num, detection in enumerate(detections):
            pose_tracks.append({})

            # Get the locked shooter ID for this frame
            shooter_id = frame_to_shooter.get(frame_num)
            if shooter_id is None:
                continue

            # Get shooter's bbox from player_tracks
            if frame_num >= len(player_tracks):
                continue

            shooter_data = player_tracks[frame_num].get(shooter_id, {})
            if "bbox" not in shooter_data:
                continue

            shooter_bbox = shooter_data["bbox"]

            # Find pose detection that matches shooter bbox
            if detection.keypoints is not None and len(detection.keypoints) > 0:
                pose_data = self.match_pose_to_bbox(
                    detection.keypoints.data.cpu().numpy(),
                    detection.boxes.xyxy.cpu().numpy(),
                    shooter_bbox
                )

                if pose_data is not None:
                    pose_tracks[frame_num][shooter_id] = pose_data

        return pose_tracks

    def match_pose_to_bbox(self, keypoints_data, boxes, target_bbox):
        """
        Match detected pose to the target player bbox.
        Returns pose data (keypoints and confidence) for the matching detection.
        """
        target_center = np.array([
            (target_bbox[0] + target_bbox[2]) / 2,
            (target_bbox[1] + target_bbox[3]) / 2
        ])

        min_distance = float('inf')
        best_match = None

        for i, box in enumerate(boxes):
            box_center = np.array([
                (box[0] + box[2]) / 2,
                (box[1] + box[3]) / 2
            ])

            distance = np.linalg.norm(target_center - box_center)
            if distance < min_distance:
                min_distance = distance
                best_match = i

        # Only accept match if it's reasonably close (within 100 pixels)
        if best_match is not None and min_distance < 100:
            # keypoints_data shape: [num_detections, num_keypoints, 3]
            # where 3 = [x, y, confidence]
            keypoints = keypoints_data[best_match]
            return {
                'keypoints': keypoints[:, :2].tolist(),  # x, y coordinates
                'confidence': keypoints[:, 2].tolist()   # confidence scores
            }

        return None
