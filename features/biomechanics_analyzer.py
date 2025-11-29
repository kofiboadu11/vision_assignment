import numpy as np
import math

class BiomechanicsAnalyzer:
    """
    Analyzes shooter's biomechanics from pose keypoints.
    Extracts features like elbow angle, knee bend, release height, etc.

    COCO 17 keypoints format:
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """

    def __init__(self, min_confidence=0.3):
        self.min_confidence = min_confidence

    def calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points.
        point2 is the vertex of the angle.
        Returns angle in degrees.
        """
        # Convert to numpy arrays
        p1 = np.array(point1)
        p2 = np.array(point2)
        p3 = np.array(point3)

        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))

        return angle

    def detect_shooting_arm(self, keypoints, confidence):
        """
        Detect which arm is the shooting arm based on wrist height.
        The shooting arm's wrist will typically be higher during release.
        Returns 'left' or 'right'
        """
        left_wrist_idx = 9
        right_wrist_idx = 10

        if (confidence[left_wrist_idx] > self.min_confidence and
            confidence[right_wrist_idx] > self.min_confidence):

            left_wrist_y = keypoints[left_wrist_idx][1]
            right_wrist_y = keypoints[right_wrist_idx][1]

            # Lower y value means higher in image (inverted coordinates)
            if left_wrist_y < right_wrist_y:
                return 'left'
            else:
                return 'right'

        # Default to right if can't determine
        return 'right'

    def extract_elbow_angle(self, keypoints, confidence, arm='right'):
        """
        Calculate elbow angle at release.
        Ideal shooting form has elbow angle around 90 degrees.
        """
        if arm == 'right':
            shoulder_idx, elbow_idx, wrist_idx = 6, 8, 10
        else:
            shoulder_idx, elbow_idx, wrist_idx = 5, 7, 9

        if (confidence[shoulder_idx] > self.min_confidence and
            confidence[elbow_idx] > self.min_confidence and
            confidence[wrist_idx] > self.min_confidence):

            angle = self.calculate_angle(
                keypoints[shoulder_idx],
                keypoints[elbow_idx],
                keypoints[wrist_idx]
            )
            return angle

        return None

    def extract_knee_bend(self, keypoints, confidence, side='right'):
        """
        Calculate knee bend angle.
        Proper shooting form requires good knee bend (around 120-140 degrees).
        """
        if side == 'right':
            hip_idx, knee_idx, ankle_idx = 12, 14, 16
        else:
            hip_idx, knee_idx, ankle_idx = 11, 13, 15

        if (confidence[hip_idx] > self.min_confidence and
            confidence[knee_idx] > self.min_confidence and
            confidence[ankle_idx] > self.min_confidence):

            angle = self.calculate_angle(
                keypoints[hip_idx],
                keypoints[knee_idx],
                keypoints[ankle_idx]
            )
            return angle

        return None

    def extract_shoulder_alignment(self, keypoints, confidence):
        """
        Calculate shoulder alignment angle.
        Shoulders should be relatively level and square to the basket.
        Returns angle in degrees from horizontal.
        """
        left_shoulder_idx = 5
        right_shoulder_idx = 6

        if (confidence[left_shoulder_idx] > self.min_confidence and
            confidence[right_shoulder_idx] > self.min_confidence):

            left_shoulder = keypoints[left_shoulder_idx]
            right_shoulder = keypoints[right_shoulder_idx]

            # Calculate angle from horizontal
            dx = right_shoulder[0] - left_shoulder[0]
            dy = right_shoulder[1] - left_shoulder[1]
            angle = abs(math.degrees(math.atan2(dy, dx)))

            return angle

        return None

    def extract_release_height(self, keypoints, confidence, arm='right'):
        """
        Calculate release height (wrist height).
        Higher release points are generally better.
        """
        wrist_idx = 10 if arm == 'right' else 9

        if confidence[wrist_idx] > self.min_confidence:
            # Return y-coordinate (lower value = higher in image)
            return keypoints[wrist_idx][1]

        return None

    def extract_body_balance(self, keypoints, confidence):
        """
        Calculate body balance/symmetry.
        Measures how centered the shooter is over their base.
        Returns deviation from center in pixels.
        """
        left_ankle_idx = 15
        right_ankle_idx = 16
        nose_idx = 0

        if (confidence[left_ankle_idx] > self.min_confidence and
            confidence[right_ankle_idx] > self.min_confidence and
            confidence[nose_idx] > self.min_confidence):

            # Calculate midpoint between ankles (base center)
            base_center_x = (keypoints[left_ankle_idx][0] + keypoints[right_ankle_idx][0]) / 2

            # Get head position (nose)
            head_x = keypoints[nose_idx][0]

            # Calculate horizontal deviation
            deviation = abs(head_x - base_center_x)

            return deviation

        return None

    def extract_follow_through(self, keypoints_sequence, confidence_sequence, arm='right'):
        """
        Analyze follow-through by tracking wrist extension after release.
        Takes a sequence of keypoints (multiple frames after release).
        Returns average wrist angle change.
        """
        wrist_idx = 10 if arm == 'right' else 9
        elbow_idx = 8 if arm == 'right' else 7

        wrist_positions = []

        for keypoints, confidence in zip(keypoints_sequence, confidence_sequence):
            if (confidence[wrist_idx] > self.min_confidence and
                confidence[elbow_idx] > self.min_confidence):

                # Calculate wrist extension angle
                wrist_y = keypoints[wrist_idx][1]
                elbow_y = keypoints[elbow_idx][1]

                wrist_positions.append(wrist_y - elbow_y)

        if len(wrist_positions) > 1:
            # Measure how much wrist drops (good follow-through)
            follow_through = wrist_positions[-1] - wrist_positions[0]
            return follow_through

        return None

    def extract_all_features(self, pose_tracks, shot_start_frame):
        """
        Extract all biomechanics features for a shot.

        Args:
            pose_tracks: Pose tracking data
            shot_start_frame: Frame number where shot begins

        Returns:
            Dictionary of biomechanics features
        """
        features = {
            'elbow_angle': None,
            'knee_bend_left': None,
            'knee_bend_right': None,
            'shoulder_alignment': None,
            'release_height': None,
            'body_balance': None,
            'follow_through': None,
            'shooting_arm': None
        }

        # Get pose data at release frame
        if shot_start_frame >= len(pose_tracks):
            return features

        frame_data = pose_tracks[shot_start_frame]

        # Get the shooter's pose (first tracked person in that frame)
        if not frame_data:
            return features

        shooter_id = list(frame_data.keys())[0]
        pose_data = frame_data[shooter_id]

        keypoints = np.array(pose_data['keypoints'])
        confidence = np.array(pose_data['confidence'])

        # Detect shooting arm
        shooting_arm = self.detect_shooting_arm(keypoints, confidence)
        features['shooting_arm'] = shooting_arm

        # Extract features
        features['elbow_angle'] = self.extract_elbow_angle(keypoints, confidence, shooting_arm)
        features['knee_bend_left'] = self.extract_knee_bend(keypoints, confidence, 'left')
        features['knee_bend_right'] = self.extract_knee_bend(keypoints, confidence, 'right')
        features['shoulder_alignment'] = self.extract_shoulder_alignment(keypoints, confidence)
        features['release_height'] = self.extract_release_height(keypoints, confidence, shooting_arm)
        features['body_balance'] = self.extract_body_balance(keypoints, confidence)

        # Extract follow-through (look at next 5 frames)
        follow_through_frames = 5
        keypoints_seq = []
        confidence_seq = []

        for i in range(shot_start_frame, min(shot_start_frame + follow_through_frames, len(pose_tracks))):
            if i < len(pose_tracks) and shooter_id in pose_tracks[i]:
                kp_data = pose_tracks[i][shooter_id]
                keypoints_seq.append(np.array(kp_data['keypoints']))
                confidence_seq.append(np.array(kp_data['confidence']))

        if len(keypoints_seq) > 1:
            features['follow_through'] = self.extract_follow_through(
                keypoints_seq, confidence_seq, shooting_arm
            )

        return features
