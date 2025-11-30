import numpy as np
from typing import Dict, List, Optional, Tuple

class BiomechanicsAnalyzer:
    """
    Analyzes shooting form biomechanics from pose keypoints.
    Uses COCO 17 keypoints format from YOLO pose estimation.

    Keypoint indices:
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """

    def __init__(self, min_confidence: float = 0.3):
        """
        Args:
            min_confidence: Minimum keypoint confidence threshold
        """
        self.min_confidence = min_confidence

        # Optimal shooting form ranges (in degrees)
        self.OPTIMAL_ELBOW_ANGLE = (85, 95)  # Degrees
        self.OPTIMAL_KNEE_BEND = (30, 60)     # Degrees from vertical
        self.MIN_RELEASE_HEIGHT_RATIO = 1.3   # Release height / shoulder height

    def analyze_shooting_form(self, pose_tracks: List[Dict],
                             release_frame: int,
                             shooting_side: str = 'right') -> Dict:
        """
        Analyze shooting form at the moment of release.

        Args:
            pose_tracks: List of pose data per frame
            release_frame: Frame index where shot is released
            shooting_side: 'left' or 'right' - which hand is shooting

        Returns:
            Dictionary of biomechanical features
        """
        if release_frame >= len(pose_tracks):
            return self._get_default_features()

        pose_data = pose_tracks[release_frame]

        if not pose_data:
            return self._get_default_features()

        # Get the shooter's pose data (assuming only one tracked player)
        shooter_id = list(pose_data.keys())[0] if pose_data else None
        if shooter_id is None:
            return self._get_default_features()

        keypoints = np.array(pose_data[shooter_id]['keypoints'])
        confidence = np.array(pose_data[shooter_id]['confidence'])

        # Determine keypoint indices based on shooting side
        if shooting_side == 'right':
            shoulder_idx, elbow_idx, wrist_idx = 6, 8, 10
            hip_idx, knee_idx, ankle_idx = 12, 14, 16
        else:
            shoulder_idx, elbow_idx, wrist_idx = 5, 7, 9
            hip_idx, knee_idx, ankle_idx = 11, 13, 15

        features = {}

        # 1. Elbow angle at release
        features['elbow_angle'] = self._calculate_elbow_angle(
            keypoints, confidence, shoulder_idx, elbow_idx, wrist_idx
        )

        # 2. Knee bend
        features['knee_bend'] = self._calculate_knee_bend(
            keypoints, confidence, hip_idx, knee_idx, ankle_idx
        )

        # 3. Release height
        features['release_height_ratio'] = self._calculate_release_height_ratio(
            keypoints, confidence, shoulder_idx, wrist_idx
        )

        # 4. Shoulder alignment (both shoulders visible)
        features['shoulder_alignment'] = self._calculate_shoulder_alignment(
            keypoints, confidence
        )

        # 5. Body balance (hips to shoulders alignment)
        features['body_balance'] = self._calculate_body_balance(
            keypoints, confidence
        )

        return features

    def _calculate_elbow_angle(self, keypoints: np.ndarray, confidence: np.ndarray,
                               shoulder_idx: int, elbow_idx: int, wrist_idx: int) -> Optional[float]:
        """Calculate angle at elbow joint (shoulder-elbow-wrist)."""
        if (confidence[shoulder_idx] < self.min_confidence or
            confidence[elbow_idx] < self.min_confidence or
            confidence[wrist_idx] < self.min_confidence):
            return None

        shoulder = keypoints[shoulder_idx]
        elbow = keypoints[elbow_idx]
        wrist = keypoints[wrist_idx]

        return self._calculate_angle(shoulder, elbow, wrist)

    def _calculate_knee_bend(self, keypoints: np.ndarray, confidence: np.ndarray,
                            hip_idx: int, knee_idx: int, ankle_idx: int) -> Optional[float]:
        """Calculate knee bend angle from vertical."""
        if (confidence[hip_idx] < self.min_confidence or
            confidence[knee_idx] < self.min_confidence or
            confidence[ankle_idx] < self.min_confidence):
            return None

        hip = keypoints[hip_idx]
        knee = keypoints[knee_idx]
        ankle = keypoints[ankle_idx]

        angle = self._calculate_angle(hip, knee, ankle)
        # Convert to bend from vertical (180 - angle)
        return 180 - angle if angle is not None else None

    def _calculate_release_height_ratio(self, keypoints: np.ndarray, confidence: np.ndarray,
                                       shoulder_idx: int, wrist_idx: int) -> Optional[float]:
        """Calculate ratio of release height to shoulder height."""
        if (confidence[shoulder_idx] < self.min_confidence or
            confidence[wrist_idx] < self.min_confidence):
            return None

        shoulder_y = keypoints[shoulder_idx][1]
        wrist_y = keypoints[wrist_idx][1]

        # In image coordinates, smaller Y = higher position
        # So ratio should be shoulder_y / wrist_y (both positive values)
        if wrist_y > 0 and shoulder_y > 0:
            # If wrist is above shoulder, ratio > 1
            return shoulder_y / wrist_y
        return None

    def _calculate_shoulder_alignment(self, keypoints: np.ndarray,
                                     confidence: np.ndarray) -> Optional[float]:
        """Calculate shoulder alignment (should be horizontal)."""
        left_shoulder_idx, right_shoulder_idx = 5, 6

        if (confidence[left_shoulder_idx] < self.min_confidence or
            confidence[right_shoulder_idx] < self.min_confidence):
            return None

        left_shoulder = keypoints[left_shoulder_idx]
        right_shoulder = keypoints[right_shoulder_idx]

        # Calculate angle from horizontal
        dy = right_shoulder[1] - left_shoulder[1]
        dx = right_shoulder[0] - left_shoulder[0]

        if dx == 0:
            return None

        angle = abs(np.degrees(np.arctan(dy / dx)))
        return angle  # Smaller angle = better alignment

    def _calculate_body_balance(self, keypoints: np.ndarray,
                               confidence: np.ndarray) -> Optional[float]:
        """Calculate body balance (vertical alignment of center of mass)."""
        # Check if all required keypoints are visible
        required_indices = [5, 6, 11, 12]  # Both shoulders and hips
        if any(confidence[idx] < self.min_confidence for idx in required_indices):
            return None

        # Calculate center of shoulders and center of hips
        shoulder_center_x = (keypoints[5][0] + keypoints[6][0]) / 2
        hip_center_x = (keypoints[11][0] + keypoints[12][0]) / 2

        # Good balance = small horizontal displacement
        displacement = abs(shoulder_center_x - hip_center_x)
        return displacement

    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[float]:
        """
        Calculate angle at point p2 formed by p1-p2-p3.
        Returns angle in degrees.
        """
        # Vectors from p2 to p1 and p2 to p3
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
        angle = np.degrees(np.arccos(cos_angle))

        return float(angle)

    def score_biomechanics(self, features: Dict) -> float:
        """
        Score biomechanics features (0-100).
        Higher score = better shooting form.
        """
        scores = []

        # 1. Elbow angle score (40% weight)
        if features.get('elbow_angle') is not None:
            elbow_angle = features['elbow_angle']
            if self.OPTIMAL_ELBOW_ANGLE[0] <= elbow_angle <= self.OPTIMAL_ELBOW_ANGLE[1]:
                elbow_score = 100
            else:
                # Penalize deviation from optimal range
                deviation = min(
                    abs(elbow_angle - self.OPTIMAL_ELBOW_ANGLE[0]),
                    abs(elbow_angle - self.OPTIMAL_ELBOW_ANGLE[1])
                )
                elbow_score = max(0, 100 - (deviation * 2))
            scores.append(('elbow', elbow_score, 0.4))

        # 2. Knee bend score (20% weight)
        if features.get('knee_bend') is not None:
            knee_bend = features['knee_bend']
            if self.OPTIMAL_KNEE_BEND[0] <= knee_bend <= self.OPTIMAL_KNEE_BEND[1]:
                knee_score = 100
            else:
                deviation = min(
                    abs(knee_bend - self.OPTIMAL_KNEE_BEND[0]),
                    abs(knee_bend - self.OPTIMAL_KNEE_BEND[1])
                )
                knee_score = max(0, 100 - (deviation * 1.5))
            scores.append(('knee', knee_score, 0.2))

        # 3. Release height score (20% weight)
        if features.get('release_height_ratio') is not None:
            height_ratio = features['release_height_ratio']
            if height_ratio >= self.MIN_RELEASE_HEIGHT_RATIO:
                height_score = 100
            else:
                height_score = (height_ratio / self.MIN_RELEASE_HEIGHT_RATIO) * 100
            scores.append(('height', height_score, 0.2))

        # 4. Shoulder alignment score (10% weight)
        if features.get('shoulder_alignment') is not None:
            alignment = features['shoulder_alignment']
            # Smaller angle is better (0-10 degrees is good)
            alignment_score = max(0, 100 - (alignment * 10))
            scores.append(('alignment', alignment_score, 0.1))

        # 5. Body balance score (10% weight)
        if features.get('body_balance') is not None:
            balance = features['body_balance']
            # Smaller displacement is better (0-20 pixels is good)
            balance_score = max(0, 100 - (balance * 5))
            scores.append(('balance', balance_score, 0.1))

        if not scores:
            return 0.0

        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in scores)
        total_weight = sum(weight for _, _, weight in scores)

        return total_score / total_weight if total_weight > 0 else 0.0

    def _get_default_features(self) -> Dict:
        """Return default features when pose data is unavailable."""
        return {
            'elbow_angle': None,
            'knee_bend': None,
            'release_height_ratio': None,
            'shoulder_alignment': None,
            'body_balance': None
        }
