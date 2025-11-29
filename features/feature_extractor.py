import numpy as np
from .biomechanics_analyzer import BiomechanicsAnalyzer
from .trajectory_analyzer import TrajectoryAnalyzer
from .contextual_analyzer import ContextualAnalyzer

class ShotFeatureExtractor:
    """
    Main feature extraction coordinator.
    Combines biomechanics, trajectory, and contextual features into a unified feature vector.
    """

    def __init__(self):
        self.biomechanics_analyzer = BiomechanicsAnalyzer()
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.contextual_analyzer = ContextualAnalyzer()

    def extract_shot_features(self, pose_tracks, ball_tracks, hoop_tracks, player_tracks,
                             shot_start_frame, shot_end_frame, shooter_id,
                             frame_width, frame_height):
        """
        Extract all features for a single shot.

        Args:
            pose_tracks: Pose tracking data
            ball_tracks: Ball tracking data
            hoop_tracks: Hoop tracking data
            player_tracks: Player tracking data
            shot_start_frame: Frame where shot begins
            shot_end_frame: Frame where shot ends (at hoop)
            shooter_id: Track ID of the shooter
            frame_width: Video frame width
            frame_height: Video frame height

        Returns:
            Dictionary containing all features
        """
        # Extract biomechanics features
        biomechanics_features = self.biomechanics_analyzer.extract_all_features(
            pose_tracks, shot_start_frame
        )

        # Extract trajectory features
        trajectory_features = self.trajectory_analyzer.extract_all_features(
            ball_tracks, hoop_tracks, shot_start_frame, shot_end_frame
        )

        # Extract contextual features
        contextual_features = self.contextual_analyzer.extract_all_features(
            shooter_id, player_tracks, hoop_tracks,
            shot_start_frame, frame_width, frame_height
        )

        # Combine all features
        all_features = {
            'biomechanics': biomechanics_features,
            'trajectory': trajectory_features,
            'contextual': contextual_features,
            'metadata': {
                'shot_start_frame': shot_start_frame,
                'shot_end_frame': shot_end_frame,
                'shooter_id': shooter_id
            }
        }

        return all_features

    def features_to_vector(self, features):
        """
        Convert feature dictionary to a numpy vector for ML model input.

        Args:
            features: Dictionary of features from extract_shot_features

        Returns:
            numpy array of feature values
        """
        feature_vector = []

        # Biomechanics features
        bio = features['biomechanics']
        feature_vector.extend([
            bio.get('elbow_angle', 0) or 0,
            bio.get('knee_bend_left', 0) or 0,
            bio.get('knee_bend_right', 0) or 0,
            bio.get('shoulder_alignment', 0) or 0,
            bio.get('release_height', 0) or 0,
            bio.get('body_balance', 0) or 0,
            bio.get('follow_through', 0) or 0,
            1 if bio.get('shooting_arm') == 'right' else 0
        ])

        # Trajectory features
        traj = features['trajectory']
        feature_vector.extend([
            traj.get('arc_height', 0) or 0,
            traj.get('release_angle', 0) or 0,
            traj.get('entry_angle', 0) or 0,
            traj.get('release_velocity', 0) or 0,
            traj.get('trajectory_smoothness', 0) or 0,
            traj.get('parabola_fit', 0) or 0,
            traj.get('peak_timing', 0) or 0
        ])

        # Contextual features
        ctx = features['contextual']
        feature_vector.extend([
            ctx.get('distance_to_basket', 0) or 0,
            ctx.get('horizontal_distance', 0) or 0,
            ctx.get('defender_distance', 0) or 0,
            ctx.get('defender_pressure', 0) or 0,
            ctx.get('shooting_angle', 0) or 0,
            ctx.get('num_defenders_nearby', 0) or 0
        ])

        # One-hot encode shot type
        shot_type = ctx.get('shot_type')
        feature_vector.extend([
            1 if shot_type == 'free_throw' else 0,
            1 if shot_type == '2pt' else 0,
            1 if shot_type == '3pt' else 0
        ])

        return np.array(feature_vector, dtype=np.float32)

    def get_feature_names(self):
        """
        Return list of feature names in the order they appear in the vector.
        """
        return [
            # Biomechanics (8 features)
            'elbow_angle',
            'knee_bend_left',
            'knee_bend_right',
            'shoulder_alignment',
            'release_height',
            'body_balance',
            'follow_through',
            'shooting_arm_right',
            # Trajectory (7 features)
            'arc_height',
            'release_angle',
            'entry_angle',
            'release_velocity',
            'trajectory_smoothness',
            'parabola_fit',
            'peak_timing',
            # Contextual (6 features)
            'distance_to_basket',
            'horizontal_distance',
            'defender_distance',
            'defender_pressure',
            'shooting_angle',
            'num_defenders_nearby',
            # Shot type (3 features - one-hot)
            'is_free_throw',
            'is_2pt',
            'is_3pt'
        ]

    def normalize_features(self, feature_vector):
        """
        Normalize feature vector for ML model input.
        This is a simple normalization - in production, use fit/transform from training data.

        Args:
            feature_vector: numpy array of features

        Returns:
            Normalized feature vector
        """
        # Clone the vector
        normalized = feature_vector.copy()

        # Define normalization ranges for each feature group
        # These are approximate ranges - adjust based on your data

        # Biomechanics (indices 0-7)
        # Angles are typically 0-180, heights are pixel values
        normalized[0] = normalized[0] / 180.0  # elbow_angle
        normalized[1] = normalized[1] / 180.0  # knee_bend_left
        normalized[2] = normalized[2] / 180.0  # knee_bend_right
        normalized[3] = normalized[3] / 180.0  # shoulder_alignment
        normalized[4] = normalized[4] / 720.0  # release_height (assuming 720p video)
        normalized[5] = normalized[5] / 100.0  # body_balance
        normalized[6] = normalized[6] / 100.0  # follow_through
        # shooting_arm_right is already 0 or 1

        # Trajectory (indices 8-14)
        normalized[8] = np.clip(normalized[8] / 500.0, 0, 1)   # arc_height
        normalized[9] = normalized[9] / 90.0    # release_angle
        normalized[10] = normalized[10] / 90.0  # entry_angle
        normalized[11] = np.clip(normalized[11] / 1000.0, 0, 1)  # release_velocity
        normalized[12] = np.clip(normalized[12], 0, 1)  # trajectory_smoothness
        # parabola_fit is already 0-1
        normalized[14] = normalized[14] / 100.0  # peak_timing

        # Contextual (indices 15-20)
        normalized[15] = np.clip(normalized[15] / 1000.0, 0, 1)  # distance_to_basket
        normalized[16] = np.clip(normalized[16] / 1000.0, 0, 1)  # horizontal_distance
        normalized[17] = np.clip(normalized[17] / 500.0, 0, 1)   # defender_distance
        # defender_pressure is already 0-1
        normalized[19] = normalized[19] / 180.0  # shooting_angle
        normalized[20] = np.clip(normalized[20] / 5.0, 0, 1)  # num_defenders_nearby

        # Shot type (indices 21-23) are already 0 or 1

        return normalized
