import json
from typing import Dict, List, Optional, Tuple
from .biomechanics_analyzer import BiomechanicsAnalyzer
from .trajectory_analyzer import TrajectoryAnalyzer

class ShotQualityAnalyzer:
    """
    Main analyzer that combines biomechanics and trajectory analysis
    to estimate overall shot quality.
    """

    def __init__(self):
        """Initialize the shot quality analyzer with sub-analyzers."""
        self.biomechanics_analyzer = BiomechanicsAnalyzer()
        self.trajectory_analyzer = TrajectoryAnalyzer()

        # Weighting for overall score
        self.BIOMECHANICS_WEIGHT = 0.4
        self.TRAJECTORY_WEIGHT = 0.5
        self.CONTEXT_WEIGHT = 0.1

        # Quality thresholds
        self.GOOD_SHOT_THRESHOLD = 70.0

    def analyze_shot(self,
                    shot_frame: int,
                    ball_tracks: List[Dict],
                    hoop_tracks: List[Dict],
                    pose_tracks: List[Dict],
                    shot_result: Optional[str] = None,
                    video_path: str = "") -> Dict:
        """
        Perform complete shot quality analysis.

        Args:
            shot_frame: Frame number where shot was detected
            ball_tracks: Ball tracking data
            hoop_tracks: Hoop tracking data
            pose_tracks: Pose tracking data
            shot_result: 'made' or 'missed' (if known)
            video_path: Path to video (used to infer shot type)

        Returns:
            Complete analysis dictionary with scores and features
        """
        # Extract trajectory (90 frames leading up to shot)
        trajectory_length = 90
        start_frame = max(0, shot_frame - trajectory_length)
        ball_trajectory = self._extract_trajectory(ball_tracks, start_frame, shot_frame + 1)

        # Get hoop position
        hoop_position = self._get_hoop_position(hoop_tracks, shot_frame)

        # Find release frame (first point of upward movement in trajectory)
        release_frame = self._find_release_frame(ball_trajectory, start_frame)

        # 1. Analyze biomechanics
        biomech_features = self.biomechanics_analyzer.analyze_shooting_form(
            pose_tracks,
            release_frame,
            shooting_side='right'  # Could be detected automatically
        )
        biomech_score = self.biomechanics_analyzer.score_biomechanics(biomech_features)

        # 2. Analyze trajectory
        trajectory_features = self.trajectory_analyzer.analyze_trajectory(
            ball_trajectory,
            hoop_position,
            release_frame_idx=max(0, release_frame - start_frame)
        )
        trajectory_score = self.trajectory_analyzer.score_trajectory(trajectory_features)

        # 3. Analyze context
        context_features = self._analyze_context(video_path, shot_frame)
        context_score = self._score_context(context_features)

        # 4. Calculate overall quality score
        overall_score = (
            self.BIOMECHANICS_WEIGHT * biomech_score +
            self.TRAJECTORY_WEIGHT * trajectory_score +
            self.CONTEXT_WEIGHT * context_score
        )

        # 5. Determine quality classification
        quality = "GOOD" if overall_score >= self.GOOD_SHOT_THRESHOLD else "BAD"

        # Compile complete analysis
        analysis = {
            'frame': shot_frame,
            'release_frame': release_frame,
            'result': shot_result,
            'quality': quality,
            'overall_score': round(overall_score, 2),
            'scores': {
                'biomechanics': round(biomech_score, 2),
                'trajectory': round(trajectory_score, 2),
                'context': round(context_score, 2)
            },
            'features': {
                'biomechanics': biomech_features,
                'trajectory': trajectory_features,
                'context': context_features
            }
        }

        return analysis

    def _extract_trajectory(self, ball_tracks: List[Dict],
                           start_frame: int, end_frame: int) -> List[Tuple[int, int]]:
        """Extract ball trajectory as list of (x, y) coordinates."""
        trajectory = []

        for frame_idx in range(start_frame, min(end_frame, len(ball_tracks))):
            ball_data = ball_tracks[frame_idx].get(1, {})
            if "bbox" in ball_data:
                bbox = ball_data["bbox"]
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                trajectory.append((center_x, center_y))

        return trajectory

    def _get_hoop_position(self, hoop_tracks: List[Dict],
                          frame_idx: int) -> Optional[Tuple[int, int]]:
        """Get hoop center position at given frame."""
        if frame_idx >= len(hoop_tracks):
            return None

        hoop_data = hoop_tracks[frame_idx].get(1, {})
        if "bbox" not in hoop_data:
            return None

        bbox = hoop_data["bbox"]
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)

        return (center_x, center_y)

    def _find_release_frame(self, trajectory: List[Tuple[int, int]],
                           start_frame: int) -> int:
        """
        Find the release frame by detecting first significant upward movement.
        Returns absolute frame number.
        """
        if len(trajectory) < 3:
            return start_frame

        # Look for first point where ball moves upward significantly
        for i in range(1, len(trajectory)):
            y_current = trajectory[i][1]
            y_prev = trajectory[i-1][1]

            # In image coords, negative change = upward movement
            if y_prev - y_current > 5:  # Moved up by more than 5 pixels
                return start_frame + i

        # If no clear release found, use first frame
        return start_frame

    def _analyze_context(self, video_path: str, frame_num: int) -> Dict:
        """Analyze contextual features of the shot."""
        # Extract shot type from video path
        shot_type = self._infer_shot_type(video_path)

        context = {
            'shot_type': shot_type,
            'frame_number': frame_num
        }

        return context

    def _infer_shot_type(self, video_path: str) -> str:
        """Infer shot type from video path."""
        video_path_lower = video_path.lower()

        if '3p' in video_path_lower:
            return '3-point'
        elif '2p' in video_path_lower:
            return '2-point'
        elif 'ft' in video_path_lower:
            return 'free-throw'
        elif 'mp' in video_path_lower:
            return 'mid-range'
        else:
            return 'unknown'

    def _score_context(self, context_features: Dict) -> float:
        """
        Score contextual features.
        For now, just returns a baseline score based on shot type.
        """
        shot_type = context_features.get('shot_type', 'unknown')

        # Free throws should have better form (more controlled)
        if shot_type == 'free-throw':
            return 80.0
        elif shot_type in ['2-point', 'mid-range']:
            return 70.0
        elif shot_type == '3-point':
            return 65.0  # Harder shots
        else:
            return 60.0

    def save_analysis(self, analyses: List[Dict], output_path: str):
        """Save shot analyses to JSON file."""
        output = {
            'shots': analyses,
            'summary': {
                'total_shots': len(analyses),
                'good_shots': sum(1 for a in analyses if a['quality'] == 'GOOD'),
                'bad_shots': sum(1 for a in analyses if a['quality'] == 'BAD'),
                'average_score': sum(a['overall_score'] for a in analyses) / len(analyses) if analyses else 0
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Shot quality analysis saved to: {output_path}")
