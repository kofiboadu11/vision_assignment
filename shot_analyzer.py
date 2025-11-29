"""
Simple shot quality analyzer focusing on key metrics only.

Analyzes 5 core factors:
1. Release angle (trajectory)
2. Arc height (trajectory)
3. Distance to basket (context)
4. Elbow angle (biomechanics)
5. Release height (biomechanics)
"""

import numpy as np
import math

class SimpleShotAnalyzer:
    """
    Lightweight shot quality analyzer with only essential metrics.
    """

    def __init__(self):
        # Ideal values for shot quality
        self.ideal_release_angle = 50  # degrees
        self.ideal_arc_height = 200    # pixels
        self.ideal_elbow_angle = 90    # degrees

    def analyze_shot(self, pose_tracks, ball_tracks, hoop_tracks, player_tracks,
                     shot_frame, frame_width, frame_height):
        """
        Analyze a single shot and return quality metrics.

        Args:
            pose_tracks: Pose tracking data
            ball_tracks: Ball tracking data
            hoop_tracks: Hoop tracking data
            player_tracks: Player tracking data
            shot_frame: Frame where shot occurs
            frame_width: Video width
            frame_height: Video height

        Returns:
            Dictionary with quality score and key metrics
        """
        # Get shooter ID
        shooter_id = None
        if shot_frame < len(pose_tracks) and pose_tracks[shot_frame]:
            shooter_id = list(pose_tracks[shot_frame].keys())[0]

        if shooter_id is None:
            return None

        # Extract the 5 key metrics
        metrics = {
            'release_angle': self._get_release_angle(ball_tracks, shot_frame),
            'arc_height': self._get_arc_height(ball_tracks, shot_frame),
            'distance_to_basket': self._get_distance(player_tracks, hoop_tracks,
                                                     shot_frame, shooter_id),
            'elbow_angle': self._get_elbow_angle(pose_tracks, shot_frame, shooter_id),
            'release_height': self._get_release_height(pose_tracks, shot_frame, shooter_id)
        }

        # Calculate quality score
        quality_score = self._calculate_quality(metrics, frame_width)

        return {
            'quality_score': quality_score,
            'metrics': metrics
        }

    def _get_release_angle(self, ball_tracks, shot_frame):
        """Calculate release angle from ball trajectory."""
        start = max(0, shot_frame - 5)
        end = min(shot_frame + 5, len(ball_tracks))

        points = []
        for i in range(start, end):
            ball_data = ball_tracks[i].get(1, {})
            if "bbox" in ball_data:
                bbox = ball_data["bbox"]
                x = (bbox[0] + bbox[2]) / 2
                y = (bbox[1] + bbox[3]) / 2
                points.append((x, y))

        if len(points) < 3:
            return None

        # Use first and last points for angle
        p1, p2 = points[0], points[-1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]  # negative dy = upward

        angle = abs(math.degrees(math.atan2(-dy, dx)))
        return angle

    def _get_arc_height(self, ball_tracks, shot_frame):
        """Calculate arc height from trajectory."""
        start = max(0, shot_frame - 30)
        end = shot_frame

        y_positions = []
        for i in range(start, end):
            ball_data = ball_tracks[i].get(1, {})
            if "bbox" in ball_data:
                bbox = ball_data["bbox"]
                y = (bbox[1] + bbox[3]) / 2
                y_positions.append(y)

        if len(y_positions) < 3:
            return None

        # Arc height is difference between start and peak
        start_y = y_positions[0]
        peak_y = min(y_positions)  # lowest y = highest point
        arc_height = start_y - peak_y

        return arc_height

    def _get_distance(self, player_tracks, hoop_tracks, frame_num, shooter_id):
        """Calculate distance from shooter to basket."""
        if frame_num >= len(player_tracks) or frame_num >= len(hoop_tracks):
            return None

        # Get shooter position
        shooter_data = player_tracks[frame_num].get(shooter_id, {})
        if "bbox" not in shooter_data:
            return None

        shooter_bbox = shooter_data["bbox"]
        shooter_x = (shooter_bbox[0] + shooter_bbox[2]) / 2
        shooter_y = (shooter_bbox[1] + shooter_bbox[3]) / 2

        # Get hoop position
        hoop_data = hoop_tracks[frame_num].get(1, {})
        if "bbox" not in hoop_data:
            return None

        hoop_bbox = hoop_data["bbox"]
        hoop_x = (hoop_bbox[0] + hoop_bbox[2]) / 2
        hoop_y = (hoop_bbox[1] + hoop_bbox[3]) / 2

        # Calculate distance
        distance = math.sqrt((hoop_x - shooter_x)**2 + (hoop_y - shooter_y)**2)
        return distance

    def _get_elbow_angle(self, pose_tracks, frame_num, shooter_id):
        """Calculate elbow angle from pose."""
        if frame_num >= len(pose_tracks):
            return None

        pose_data = pose_tracks[frame_num].get(shooter_id, {})
        if not pose_data:
            return None

        keypoints = np.array(pose_data['keypoints'])
        confidence = np.array(pose_data['confidence'])

        # Detect shooting arm (higher wrist)
        left_wrist_y = keypoints[9][1] if confidence[9] > 0.3 else float('inf')
        right_wrist_y = keypoints[10][1] if confidence[10] > 0.3 else float('inf')

        if left_wrist_y < right_wrist_y:
            shoulder_idx, elbow_idx, wrist_idx = 5, 7, 9
        else:
            shoulder_idx, elbow_idx, wrist_idx = 6, 8, 10

        # Check confidence
        if (confidence[shoulder_idx] < 0.3 or
            confidence[elbow_idx] < 0.3 or
            confidence[wrist_idx] < 0.3):
            return None

        # Calculate angle
        p1 = keypoints[shoulder_idx]
        p2 = keypoints[elbow_idx]
        p3 = keypoints[wrist_idx]

        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))

        return angle

    def _get_release_height(self, pose_tracks, frame_num, shooter_id):
        """Get release height (wrist y-coordinate)."""
        if frame_num >= len(pose_tracks):
            return None

        pose_data = pose_tracks[frame_num].get(shooter_id, {})
        if not pose_data:
            return None

        keypoints = np.array(pose_data['keypoints'])
        confidence = np.array(pose_data['confidence'])

        # Get higher wrist (shooting hand)
        left_wrist_y = keypoints[9][1] if confidence[9] > 0.3 else float('inf')
        right_wrist_y = keypoints[10][1] if confidence[10] > 0.3 else float('inf')

        release_height = min(left_wrist_y, right_wrist_y)

        return release_height if release_height != float('inf') else None

    def _calculate_quality(self, metrics, frame_width):
        """
        Calculate overall quality score from metrics.
        Simple weighted average of deviations from ideal.
        """
        scores = []

        # 1. Release angle (weight: 25%)
        if metrics['release_angle'] is not None:
            deviation = abs(metrics['release_angle'] - self.ideal_release_angle)
            score = max(0, 100 - deviation * 2)  # -2 points per degree off
            scores.append(score * 0.25)

        # 2. Arc height (weight: 25%)
        if metrics['arc_height'] is not None:
            deviation = abs(metrics['arc_height'] - self.ideal_arc_height)
            score = max(0, 100 - (deviation / 2))  # -0.5 points per pixel off
            scores.append(score * 0.25)

        # 3. Distance penalty (weight: 20%)
        if metrics['distance_to_basket'] is not None:
            normalized_dist = metrics['distance_to_basket'] / frame_width
            penalty = normalized_dist * 100  # farther = lower score
            score = max(0, 100 - penalty)
            scores.append(score * 0.20)

        # 4. Elbow angle (weight: 15%)
        if metrics['elbow_angle'] is not None:
            deviation = abs(metrics['elbow_angle'] - self.ideal_elbow_angle)
            score = max(0, 100 - deviation * 1.5)
            scores.append(score * 0.15)

        # 5. Release height (weight: 15%)
        if metrics['release_height'] is not None:
            # Lower y = higher position = better (inverted coordinates)
            # Normalize by frame height, ideal is upper third
            normalized_height = metrics['release_height'] / 720  # assume 720p
            score = max(0, 100 - normalized_height * 50)
            scores.append(score * 0.15)

        if len(scores) == 0:
            return 50.0  # neutral score if no metrics available

        # Overall quality
        quality = sum(scores) / (sum([0.25, 0.25, 0.20, 0.15, 0.15]) * len(scores) / 5)

        return round(quality, 1)
