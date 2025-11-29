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

    def get_pose_tracks(self, frames, player_tracks, ball_tracks):
        """
        Get pose keypoints for the shooter in each frame.
        Returns a list of dicts with pose data per frame.
        Format: [{track_id: {'keypoints': [...], 'confidence': [...]}}, ...]
        """
        detections = self.detect_frames(frames)
        pose_tracks = []

        for frame_num, detection in enumerate(detections):
            pose_tracks.append({})

            # Identify the shooter for this frame
            shooter_id = self.get_shooter_id(player_tracks, ball_tracks, frame_num)
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
