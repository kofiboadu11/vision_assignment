import numpy as np
import math

class ContextualAnalyzer:
    """
    Analyzes contextual features that affect shot quality:
    - Distance to basket
    - Defender proximity
    - Shot type classification (2pt, 3pt, free throw)
    - Shooter position
    """

    def __init__(self):
        pass

    def calculate_distance_to_basket(self, shooter_position, hoop_position):
        """
        Calculate Euclidean distance from shooter to basket.

        Args:
            shooter_position: (x, y) tuple of shooter center
            hoop_position: (x, y) tuple of hoop center

        Returns:
            Distance in pixels
        """
        if shooter_position is None or hoop_position is None:
            return None

        dx = hoop_position[0] - shooter_position[0]
        dy = hoop_position[1] - shooter_position[1]

        distance = math.sqrt(dx**2 + dy**2)

        return distance

    def calculate_horizontal_distance(self, shooter_position, hoop_position):
        """
        Calculate horizontal (x-axis) distance from shooter to basket.
        Useful for determining shot type.
        """
        if shooter_position is None or hoop_position is None:
            return None

        horizontal_dist = abs(hoop_position[0] - shooter_position[0])

        return horizontal_dist

    def classify_shot_type(self, distance, frame_width):
        """
        Classify shot as 2-point, 3-point, or free throw based on distance.

        Args:
            distance: Distance to basket in pixels
            frame_width: Video frame width for normalization

        Returns:
            'free_throw', '2pt', or '3pt'
        """
        if distance is None:
            return None

        # Normalize distance by frame width
        normalized_dist = distance / frame_width

        # These thresholds may need tuning based on camera angle/distance
        if normalized_dist < 0.15:
            return 'free_throw'
        elif normalized_dist < 0.35:
            return '2pt'
        else:
            return '3pt'

    def detect_nearest_defender(self, shooter_id, shooter_position, player_tracks, frame_num):
        """
        Find the nearest defender (other player) to the shooter.

        Args:
            shooter_id: Track ID of the shooter
            shooter_position: (x, y) tuple of shooter center
            player_tracks: Player tracking data
            frame_num: Current frame number

        Returns:
            Dictionary with defender info: {'distance': float, 'position': (x, y)}
        """
        if frame_num >= len(player_tracks):
            return None

        min_distance = float('inf')
        defender_info = None

        for track_id, player_data in player_tracks[frame_num].items():
            # Skip the shooter
            if track_id == shooter_id:
                continue

            if "bbox" not in player_data:
                continue

            bbox = player_data["bbox"]
            player_pos = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            # Calculate distance
            dist = math.sqrt(
                (player_pos[0] - shooter_position[0])**2 +
                (player_pos[1] - shooter_position[1])**2
            )

            if dist < min_distance:
                min_distance = dist
                defender_info = {
                    'distance': dist,
                    'position': player_pos,
                    'track_id': track_id
                }

        return defender_info

    def calculate_defender_pressure(self, defender_distance, frame_width):
        """
        Calculate defender pressure metric.
        Closer defenders = more pressure = harder shot.

        Args:
            defender_distance: Distance to nearest defender in pixels
            frame_width: Frame width for normalization

        Returns:
            Pressure score (0-1, higher = more pressure)
        """
        if defender_distance is None:
            return 0.0

        # Normalize by frame width
        normalized_dist = defender_distance / frame_width

        # Convert to pressure score (inverse relationship)
        # Close defender (< 0.1 normalized) = high pressure
        # Far defender (> 0.3 normalized) = low pressure
        if normalized_dist < 0.1:
            pressure = 1.0
        elif normalized_dist > 0.3:
            pressure = 0.0
        else:
            # Linear interpolation
            pressure = 1.0 - ((normalized_dist - 0.1) / 0.2)

        return pressure

    def calculate_shooting_angle(self, shooter_position, hoop_position):
        """
        Calculate angle from shooter to basket (from shooter's perspective).

        Args:
            shooter_position: (x, y) tuple of shooter center
            hoop_position: (x, y) tuple of hoop center

        Returns:
            Angle in degrees
        """
        if shooter_position is None or hoop_position is None:
            return None

        dx = hoop_position[0] - shooter_position[0]
        dy = hoop_position[1] - shooter_position[1]

        angle = math.degrees(math.atan2(dy, dx))

        return angle

    def extract_all_features(self, shooter_id, player_tracks, hoop_tracks,
                           shot_frame, frame_width, frame_height):
        """
        Extract all contextual features for a shot.

        Args:
            shooter_id: Track ID of the shooter
            player_tracks: Player tracking data
            hoop_tracks: Hoop tracking data
            shot_frame: Frame number where shot occurs
            frame_width: Video frame width
            frame_height: Video frame height

        Returns:
            Dictionary of contextual features
        """
        features = {
            'distance_to_basket': None,
            'horizontal_distance': None,
            'shot_type': None,
            'defender_distance': None,
            'defender_pressure': None,
            'shooting_angle': None,
            'num_defenders_nearby': 0
        }

        # Get shooter position
        shooter_position = None
        if shot_frame < len(player_tracks):
            shooter_data = player_tracks[shot_frame].get(shooter_id, {})
            if "bbox" in shooter_data:
                bbox = shooter_data["bbox"]
                shooter_position = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        # Get hoop position
        hoop_position = None
        if shot_frame < len(hoop_tracks):
            hoop_data = hoop_tracks[shot_frame].get(1, {})
            if "bbox" in hoop_data:
                bbox = hoop_data["bbox"]
                hoop_position = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        if shooter_position is None or hoop_position is None:
            return features

        # Calculate distance features
        features['distance_to_basket'] = self.calculate_distance_to_basket(
            shooter_position, hoop_position
        )
        features['horizontal_distance'] = self.calculate_horizontal_distance(
            shooter_position, hoop_position
        )
        features['shooting_angle'] = self.calculate_shooting_angle(
            shooter_position, hoop_position
        )

        # Classify shot type
        features['shot_type'] = self.classify_shot_type(
            features['distance_to_basket'], frame_width
        )

        # Detect defender
        defender_info = self.detect_nearest_defender(
            shooter_id, shooter_position, player_tracks, shot_frame
        )

        if defender_info:
            features['defender_distance'] = defender_info['distance']
            features['defender_pressure'] = self.calculate_defender_pressure(
                defender_info['distance'], frame_width
            )

        # Count nearby defenders (within 20% of frame width)
        nearby_threshold = frame_width * 0.2
        num_nearby = 0

        if shot_frame < len(player_tracks):
            for track_id, player_data in player_tracks[shot_frame].items():
                if track_id == shooter_id or "bbox" not in player_data:
                    continue

                bbox = player_data["bbox"]
                player_pos = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                dist = math.sqrt(
                    (player_pos[0] - shooter_position[0])**2 +
                    (player_pos[1] - shooter_position[1])**2
                )

                if dist < nearby_threshold:
                    num_nearby += 1

        features['num_defenders_nearby'] = num_nearby

        return features
