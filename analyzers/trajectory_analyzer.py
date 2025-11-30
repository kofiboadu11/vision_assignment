import numpy as np
from typing import Dict, List, Optional, Tuple

class TrajectoryAnalyzer:
    """
    Analyzes ball trajectory to extract shot quality features.
    """

    def __init__(self):
        """Initialize trajectory analyzer with optimal parameters."""
        # Optimal trajectory parameters
        self.OPTIMAL_ARC_ANGLE = (45, 55)  # Degrees
        self.OPTIMAL_ENTRY_ANGLE = (40, 50)  # Degrees
        self.MIN_ARC_HEIGHT_PIXELS = 50     # Minimum arc height above release

    def analyze_trajectory(self, ball_trajectory: List[Tuple[int, int]],
                          hoop_position: Optional[Tuple[int, int]] = None,
                          release_frame_idx: int = 0) -> Dict:
        """
        Analyze ball trajectory to extract quality features.

        Args:
            ball_trajectory: List of (x, y) ball positions
            hoop_position: (x, y) position of hoop center
            release_frame_idx: Index in trajectory where ball was released

        Returns:
            Dictionary of trajectory features
        """
        if len(ball_trajectory) < 3:
            return self._get_default_features()

        features = {}

        # 1. Arc height (peak height above release point)
        features['arc_height'] = self._calculate_arc_height(
            ball_trajectory, release_frame_idx
        )

        # 2. Release angle (initial trajectory angle)
        features['release_angle'] = self._calculate_release_angle(
            ball_trajectory, release_frame_idx
        )

        # 3. Entry angle (angle when approaching hoop)
        if hoop_position:
            features['entry_angle'] = self._calculate_entry_angle(
                ball_trajectory, hoop_position
            )
        else:
            features['entry_angle'] = None

        # 4. Shot distance (distance from release point to hoop)
        if hoop_position and release_frame_idx < len(ball_trajectory):
            features['shot_distance'] = self._calculate_shot_distance(
                ball_trajectory[release_frame_idx], hoop_position
            )
        else:
            features['shot_distance'] = None

        # 5. Trajectory smoothness (consistency of parabolic fit)
        features['trajectory_smoothness'] = self._calculate_smoothness(
            ball_trajectory
        )

        # 6. Peak frame index
        features['peak_frame_idx'] = self._find_peak_frame(ball_trajectory)

        return features

    def _calculate_arc_height(self, trajectory: List[Tuple[int, int]],
                             release_idx: int = 0) -> Optional[float]:
        """Calculate the height of the arc above the release point."""
        if release_idx >= len(trajectory):
            return None

        release_y = trajectory[release_idx][1]

        # Find the highest point (minimum y in image coordinates)
        min_y = min(point[1] for point in trajectory)

        # Arc height is the difference (in image coords, smaller Y = higher)
        arc_height = release_y - min_y

        return float(arc_height) if arc_height > 0 else 0.0

    def _calculate_release_angle(self, trajectory: List[Tuple[int, int]],
                                release_idx: int = 0) -> Optional[float]:
        """Calculate the initial trajectory angle from horizontal."""
        # Use first few points after release to estimate initial angle
        if release_idx + 2 >= len(trajectory):
            return None

        p1 = np.array(trajectory[release_idx])
        p2 = np.array(trajectory[release_idx + 2])  # Use point 2 frames later

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]  # In image coords, negative dy = upward

        if dx == 0:
            return 90.0  # Vertical shot

        # Calculate angle from horizontal (negative dy = positive angle)
        angle = np.degrees(np.arctan(-dy / abs(dx)))

        return float(angle)

    def _calculate_entry_angle(self, trajectory: List[Tuple[int, int]],
                              hoop_position: Tuple[int, int]) -> Optional[float]:
        """Calculate angle at which ball approaches the hoop."""
        if len(trajectory) < 2:
            return None

        # Find the point closest to hoop
        hoop_x, hoop_y = hoop_position
        min_dist = float('inf')
        closest_idx = 0

        for i, (x, y) in enumerate(trajectory):
            dist = np.sqrt((x - hoop_x)**2 + (y - hoop_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Use the point before the closest point and the closest point
        if closest_idx == 0:
            return None

        p1 = np.array(trajectory[closest_idx - 1])
        p2 = np.array(trajectory[closest_idx])

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if dx == 0:
            return 90.0

        # Entry angle from horizontal (negative dy = positive angle)
        angle = np.degrees(np.arctan(-dy / abs(dx)))

        return float(abs(angle))  # Use absolute value

    def _calculate_shot_distance(self, release_point: Tuple[int, int],
                                 hoop_position: Tuple[int, int]) -> float:
        """Calculate Euclidean distance from release point to hoop."""
        release_x, release_y = release_point
        hoop_x, hoop_y = hoop_position

        distance = np.sqrt((hoop_x - release_x)**2 + (hoop_y - release_y)**2)
        return float(distance)

    def _calculate_smoothness(self, trajectory: List[Tuple[int, int]]) -> Optional[float]:
        """
        Calculate trajectory smoothness by fitting a parabola and measuring residuals.
        Lower residual = smoother trajectory.
        Returns normalized smoothness score (0-100, higher is better).
        """
        if len(trajectory) < 5:
            return None

        # Extract x and y coordinates
        x_coords = np.array([p[0] for p in trajectory])
        y_coords = np.array([p[1] for p in trajectory])

        try:
            # Fit 2nd degree polynomial (parabola)
            coeffs = np.polyfit(x_coords, y_coords, 2)
            poly = np.poly1d(coeffs)

            # Calculate predicted y values
            y_pred = poly(x_coords)

            # Calculate mean squared error
            mse = np.mean((y_coords - y_pred) ** 2)

            # Convert to smoothness score (lower MSE = higher smoothness)
            # Normalize: good trajectories have MSE < 100
            smoothness = max(0, 100 - mse)

            return float(smoothness)

        except:
            return None

    def _find_peak_frame(self, trajectory: List[Tuple[int, int]]) -> int:
        """Find the frame index where ball reaches peak height (minimum y)."""
        min_y = float('inf')
        peak_idx = 0

        for i, (x, y) in enumerate(trajectory):
            if y < min_y:
                min_y = y
                peak_idx = i

        return peak_idx

    def score_trajectory(self, features: Dict) -> float:
        """
        Score trajectory features (0-100).
        Higher score = better shot trajectory.
        """
        scores = []

        # 1. Arc height score (30% weight)
        if features.get('arc_height') is not None:
            arc_height = features['arc_height']
            if arc_height >= self.MIN_ARC_HEIGHT_PIXELS:
                # Good arc, score based on height (up to 200 pixels is optimal)
                arc_score = min(100, (arc_height / 200) * 100)
            else:
                # Too flat
                arc_score = (arc_height / self.MIN_ARC_HEIGHT_PIXELS) * 50
            scores.append(('arc_height', arc_score, 0.3))

        # 2. Release angle score (25% weight)
        if features.get('release_angle') is not None:
            release_angle = features['release_angle']
            if self.OPTIMAL_ARC_ANGLE[0] <= release_angle <= self.OPTIMAL_ARC_ANGLE[1]:
                release_score = 100
            else:
                deviation = min(
                    abs(release_angle - self.OPTIMAL_ARC_ANGLE[0]),
                    abs(release_angle - self.OPTIMAL_ARC_ANGLE[1])
                )
                release_score = max(0, 100 - (deviation * 2))
            scores.append(('release_angle', release_score, 0.25))

        # 3. Entry angle score (25% weight)
        if features.get('entry_angle') is not None:
            entry_angle = features['entry_angle']
            if self.OPTIMAL_ENTRY_ANGLE[0] <= entry_angle <= self.OPTIMAL_ENTRY_ANGLE[1]:
                entry_score = 100
            else:
                deviation = min(
                    abs(entry_angle - self.OPTIMAL_ENTRY_ANGLE[0]),
                    abs(entry_angle - self.OPTIMAL_ENTRY_ANGLE[1])
                )
                entry_score = max(0, 100 - (deviation * 2))
            scores.append(('entry_angle', entry_score, 0.25))

        # 4. Trajectory smoothness score (20% weight)
        if features.get('trajectory_smoothness') is not None:
            smoothness = features['trajectory_smoothness']
            scores.append(('smoothness', smoothness, 0.2))

        if not scores:
            return 0.0

        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in scores)
        total_weight = sum(weight for _, _, weight in scores)

        return total_score / total_weight if total_weight > 0 else 0.0

    def _get_default_features(self) -> Dict:
        """Return default features when trajectory data is unavailable."""
        return {
            'arc_height': None,
            'release_angle': None,
            'entry_angle': None,
            'shot_distance': None,
            'trajectory_smoothness': None,
            'peak_frame_idx': None
        }
