import numpy as np
import math

class TrajectoryAnalyzer:
    """
    Analyzes ball trajectory to extract shot quality features.
    Features include arc height, entry angle, release angle, velocity, etc.
    """

    def __init__(self):
        pass

    def get_trajectory_points(self, ball_tracks, start_frame, end_frame):
        """
        Extract ball center positions for a shot sequence.
        """
        trajectory = []

        for frame_idx in range(start_frame, min(end_frame + 1, len(ball_tracks))):
            ball_data = ball_tracks[frame_idx].get(1, {})
            if "bbox" in ball_data:
                bbox = ball_data["bbox"]
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                trajectory.append((center_x, center_y, frame_idx))

        return trajectory

    def calculate_arc_height(self, trajectory):
        """
        Calculate arc height (vertical distance from release to peak).
        Higher arcs are generally better for shot success.
        """
        if len(trajectory) < 3:
            return None

        # Find release point (first point)
        release_y = trajectory[0][1]

        # Find peak (minimum y, since lower y = higher in image)
        peak_y = min([pt[1] for pt in trajectory])

        # Arc height is the vertical distance
        arc_height = release_y - peak_y

        return arc_height

    def calculate_release_angle(self, trajectory):
        """
        Calculate release angle (initial trajectory angle from horizontal).
        Ideal release angle is typically 45-55 degrees.
        """
        if len(trajectory) < 5:
            return None

        # Use first few points to estimate initial angle
        p1 = trajectory[0]
        p2 = trajectory[min(4, len(trajectory) - 1)]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]  # positive dy means going down

        # Calculate angle (negative dy means going up)
        angle = math.degrees(math.atan2(-dy, dx))

        return angle

    def calculate_entry_angle(self, trajectory, hoop_position):
        """
        Calculate entry angle into the basket.
        Steeper entry angles (45-60 degrees) are better.

        Args:
            trajectory: List of (x, y, frame) tuples
            hoop_position: (x, y) tuple of hoop center
        """
        if len(trajectory) < 5 or hoop_position is None:
            return None

        # Find point closest to hoop
        hoop_x, hoop_y = hoop_position

        min_dist = float('inf')
        closest_idx = -1

        for i, (x, y, _) in enumerate(trajectory):
            dist = math.sqrt((x - hoop_x)**2 + (y - hoop_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Calculate angle from a few frames before hoop entry
        if closest_idx >= 4:
            p1 = trajectory[closest_idx - 4]
            p2 = trajectory[closest_idx]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            # Entry angle (steeper = better)
            angle = abs(math.degrees(math.atan2(dy, dx)))

            return angle

        return None

    def calculate_release_velocity(self, trajectory, fps=30):
        """
        Estimate release velocity from initial trajectory.

        Args:
            trajectory: List of (x, y, frame) tuples
            fps: Frames per second of video

        Returns:
            Velocity in pixels per second
        """
        if len(trajectory) < 3:
            return None

        # Use first few points
        p1 = trajectory[0]
        p2 = trajectory[min(2, len(trajectory) - 1)]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dt = (p2[2] - p1[2]) / fps

        if dt == 0:
            return None

        # Calculate velocity magnitude
        velocity = math.sqrt(dx**2 + dy**2) / dt

        return velocity

    def calculate_trajectory_smoothness(self, trajectory):
        """
        Calculate trajectory smoothness (consistency).
        Smoother trajectories indicate better shot mechanics.
        Returns variance in trajectory curvature.
        """
        if len(trajectory) < 5:
            return None

        # Calculate second derivatives (curvature)
        curvatures = []

        for i in range(1, len(trajectory) - 1):
            p0 = np.array(trajectory[i-1][:2])
            p1 = np.array(trajectory[i][:2])
            p2 = np.array(trajectory[i+1][:2])

            # Calculate curvature
            v1 = p1 - p0
            v2 = p2 - p1

            # Angle change
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = math.acos(np.clip(cos_angle, -1.0, 1.0))
                curvatures.append(angle)

        if len(curvatures) > 0:
            # Lower variance = smoother trajectory
            smoothness = np.var(curvatures)
            return smoothness

        return None

    def fit_parabola(self, trajectory):
        """
        Fit a parabolic curve to the trajectory.
        Returns polynomial coefficients and R-squared value.
        """
        if len(trajectory) < 5:
            return None, None

        x_coords = np.array([pt[0] for pt in trajectory])
        y_coords = np.array([pt[1] for pt in trajectory])

        try:
            # Fit 2nd degree polynomial
            coeffs = np.polyfit(x_coords, y_coords, 2)

            # Calculate R-squared
            poly = np.poly1d(coeffs)
            y_pred = poly(x_coords)
            ss_res = np.sum((y_coords - y_pred)**2)
            ss_tot = np.sum((y_coords - np.mean(y_coords))**2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-6))

            return coeffs, r_squared
        except:
            return None, None

    def extract_peak_timing(self, trajectory):
        """
        Calculate when the peak occurs relative to total flight time.
        Optimal shots typically peak around 50-60% of flight time.
        Returns percentage (0-100).
        """
        if len(trajectory) < 3:
            return None

        # Find peak index
        peak_idx = 0
        min_y = trajectory[0][1]

        for i, (x, y, _) in enumerate(trajectory):
            if y < min_y:
                min_y = y
                peak_idx = i

        # Calculate percentage
        peak_timing = (peak_idx / len(trajectory)) * 100

        return peak_timing

    def extract_all_features(self, ball_tracks, hoop_tracks, shot_start_frame, shot_end_frame):
        """
        Extract all trajectory features for a shot.

        Args:
            ball_tracks: Ball tracking data
            hoop_tracks: Hoop tracking data
            shot_start_frame: Frame where shot begins
            shot_end_frame: Frame where shot ends (at hoop)

        Returns:
            Dictionary of trajectory features
        """
        features = {
            'arc_height': None,
            'release_angle': None,
            'entry_angle': None,
            'release_velocity': None,
            'trajectory_smoothness': None,
            'parabola_fit': None,
            'peak_timing': None
        }

        # Get trajectory points
        trajectory = self.get_trajectory_points(ball_tracks, shot_start_frame, shot_end_frame)

        if len(trajectory) < 3:
            return features

        # Get hoop position
        hoop_position = None
        if shot_end_frame < len(hoop_tracks):
            hoop_data = hoop_tracks[shot_end_frame].get(1, {})
            if "bbox" in hoop_data:
                bbox = hoop_data["bbox"]
                hoop_position = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        # Extract features
        features['arc_height'] = self.calculate_arc_height(trajectory)
        features['release_angle'] = self.calculate_release_angle(trajectory)
        features['entry_angle'] = self.calculate_entry_angle(trajectory, hoop_position)
        features['release_velocity'] = self.calculate_release_velocity(trajectory)
        features['trajectory_smoothness'] = self.calculate_trajectory_smoothness(trajectory)
        features['peak_timing'] = self.extract_peak_timing(trajectory)

        # Fit parabola
        coeffs, r_squared = self.fit_parabola(trajectory)
        if r_squared is not None:
            features['parabola_fit'] = r_squared

        return features
