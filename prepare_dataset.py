"""
Dataset preparation module for Basketball-51 dataset.
Extracts features from all videos and prepares them for model training.
"""
import os
import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm

from utils import read_video
from trackers import PlayerTracker, BallTracker, HoopTracker, PoseTracker
from analyzers import ShotQualityAnalyzer


class Basketball51Dataset:
    """
    Prepares Basketball-51 dataset for machine learning.

    Dataset structure:
    - 3p0, 2p0, ft0, mp0: Missed shots (label=0)
    - 3p1, 2p1, ft1, mp1: Made shots (label=1)
    """

    def __init__(self, dataset_path: str, model_path: str):
        """
        Args:
            dataset_path: Path to Basketball-51 dataset directory
            model_path: Path to YOLO model for detection
        """
        self.dataset_path = Path(dataset_path)
        self.model_path = model_path

        # Initialize trackers
        print("Loading models...")
        self.player_tracker = PlayerTracker(model_path)
        self.ball_tracker = BallTracker(model_path)
        self.hoop_tracker = HoopTracker(model_path)
        self.pose_tracker = PoseTracker()
        self.quality_analyzer = ShotQualityAnalyzer()

    def get_video_files(self) -> List[Tuple[str, int, str]]:
        """
        Get all video files with their labels.

        Returns:
            List of (video_path, label, shot_type) tuples
        """
        video_files = []

        # Define folder patterns: name -> (label, shot_type)
        folder_mapping = {
            '3p0': (0, '3-point'),
            '3p1': (1, '3-point'),
            '2p0': (0, '2-point'),
            '2p1': (1, '2-point'),
            'ft0': (0, 'free-throw'),
            'ft1': (1, 'free-throw'),
            'mp0': (0, 'mid-range'),
            'mp1': (1, 'mid-range'),
        }

        for folder_name, (label, shot_type) in folder_mapping.items():
            folder_path = self.dataset_path / folder_name

            if not folder_path.exists():
                print(f"Warning: Folder {folder_name} not found")
                continue

            # Get all video files in folder
            for video_file in folder_path.glob('*.mp4'):
                video_files.append((str(video_file), label, shot_type))

        print(f"Found {len(video_files)} videos in dataset")
        return video_files

    def extract_features_from_video(self, video_path: str) -> Dict:
        """
        Extract features from a single video.

        Returns:
            Dictionary with extracted features or None if extraction failed
        """
        try:
            # Read video
            video_frames = read_video(video_path)

            if len(video_frames) == 0:
                return None

            # Track objects
            player_tracks = self.player_tracker.get_object_tracks(video_frames)

            ball_tracks = self.ball_tracker.get_object_tracks(video_frames)
            ball_tracks = self.ball_tracker.remove_wrong_detections(ball_tracks)
            ball_tracks = self.ball_tracker.interpolate_ball_positions(ball_tracks)

            hoop_tracks = self.hoop_tracker.get_object_tracks(video_frames)
            pose_tracks = self.pose_tracker.get_pose_tracks(video_frames, player_tracks, ball_tracks)

            # Analyze shot (assuming shot happens around middle of video)
            shot_frame = len(video_frames) // 2

            analysis = self.quality_analyzer.analyze_shot(
                shot_frame=shot_frame,
                ball_tracks=ball_tracks,
                hoop_tracks=hoop_tracks,
                pose_tracks=pose_tracks,
                shot_result=None,
                video_path=video_path
            )

            # Extract numerical features
            features = self._extract_numerical_features(analysis)

            return features

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return None

    def _extract_numerical_features(self, analysis: Dict) -> Dict:
        """Extract numerical features from analysis dictionary."""
        features = {}

        # Biomechanics features
        bio_features = analysis['features']['biomechanics']
        features['elbow_angle'] = bio_features.get('elbow_angle')
        features['knee_bend'] = bio_features.get('knee_bend')
        features['release_height_ratio'] = bio_features.get('release_height_ratio')
        features['shoulder_alignment'] = bio_features.get('shoulder_alignment')
        features['body_balance'] = bio_features.get('body_balance')

        # Trajectory features
        traj_features = analysis['features']['trajectory']
        features['arc_height'] = traj_features.get('arc_height')
        features['release_angle'] = traj_features.get('release_angle')
        features['entry_angle'] = traj_features.get('entry_angle')
        features['shot_distance'] = traj_features.get('shot_distance')
        features['trajectory_smoothness'] = traj_features.get('trajectory_smoothness')

        # Scores (already computed)
        features['biomechanics_score'] = analysis['scores']['biomechanics']
        features['trajectory_score'] = analysis['scores']['trajectory']

        # Context features (one-hot encode shot type)
        shot_type = analysis['features']['context']['shot_type']
        features['is_3point'] = 1 if shot_type == '3-point' else 0
        features['is_2point'] = 1 if shot_type == '2-point' else 0
        features['is_freethrow'] = 1 if shot_type == 'free-throw' else 0
        features['is_midrange'] = 1 if shot_type == 'mid-range' else 0

        return features

    def prepare_dataset(self, output_path: str, max_videos: int = None):
        """
        Extract features from all videos and save to file.

        Args:
            output_path: Path to save extracted features
            max_videos: Maximum number of videos to process (for testing)
        """
        video_files = self.get_video_files()

        if max_videos:
            # Ensure balanced class sampling
            made_videos = [v for v in video_files if v[1] == 1]
            missed_videos = [v for v in video_files if v[1] == 0]

            # Take equal numbers from each class
            videos_per_class = max_videos // 2
            video_files = (
                made_videos[:videos_per_class] +
                missed_videos[:videos_per_class]
            )
            print(f"Balanced sampling: {len([v for v in video_files if v[1] == 1])} made, "
                  f"{len([v for v in video_files if v[1] == 0])} missed")

        dataset = []

        print(f"\nExtracting features from {len(video_files)} videos...")
        for video_path, label, shot_type in tqdm(video_files):
            features = self.extract_features_from_video(video_path)

            if features is not None:
                dataset.append({
                    'features': features,
                    'label': label,
                    'shot_type': shot_type,
                    'video_path': video_path
                })

        # Save dataset
        print(f"\nSaving dataset to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"Dataset prepared: {len(dataset)} samples")
        print(f"Made shots: {sum(1 for d in dataset if d['label'] == 1)}")
        print(f"Missed shots: {sum(1 for d in dataset if d['label'] == 0)}")

        return dataset


if __name__ == "__main__":
    # Example usage
    dataset = Basketball51Dataset(
        dataset_path="input_videos/Basketball_51 dataset",
        model_path="basketball_predictor_V3.pt"
    )

    # Extract features from first 10 videos for testing
    dataset.prepare_dataset(
        output_path="dataset_features.json",
        max_videos=10
    )
