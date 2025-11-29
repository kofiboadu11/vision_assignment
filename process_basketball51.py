"""
Process Basketball-51 dataset videos and extract shot quality features.

Dataset structure expected:
basketball51/
├── 2pt/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── 3pt/
│   ├── video1.mp4
│   └── ...
├── free_throw/
│   └── ...
└── mid_range/
    └── ...
"""

import os
import json
import glob
from pathlib import Path
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker, HoopTracker, PoseTracker
from features import ShotFeatureExtractor
from models import ShotQualityPredictor

def detect_shots_simple(ball_tracks, hoop_tracks):
    """Simplified shot detection for dataset processing."""
    from main import detect_shots
    return detect_shots(ball_tracks, hoop_tracks)

def process_single_video(video_path, output_dir, model_path, shot_type=None):
    """
    Process a single video and extract shot features.

    Args:
        video_path: Path to video file
        output_dir: Directory to save results
        model_path: Path to YOLO model
        shot_type: Type of shot (2pt, 3pt, free_throw, mid_range)

    Returns:
        Dictionary with extracted features for all shots in video
    """
    print(f"\nProcessing: {video_path}")

    try:
        # 1. Read video
        video_frames = read_video(video_path)
        if len(video_frames) == 0:
            print(f"  ⚠️  Empty video, skipping")
            return None

        frame_height, frame_width = video_frames[0].shape[:2]

        # 2. Track objects
        print("  Tracking objects...")
        player_tracker = PlayerTracker(model_path)
        ball_tracker = BallTracker(model_path)
        hoop_tracker = HoopTracker(model_path)
        pose_tracker = PoseTracker()

        player_tracks = player_tracker.get_object_tracks(video_frames)
        ball_tracks = ball_tracker.get_object_tracks(video_frames)
        ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
        ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
        hoop_tracks = hoop_tracker.get_object_tracks(video_frames)

        # 3. Track pose
        print("  Tracking pose...")
        pose_tracks = pose_tracker.get_pose_tracks(video_frames, player_tracks, ball_tracks)

        # 4. Detect shots
        print("  Detecting shots...")
        shot_frames, shot_results = detect_shots_simple(ball_tracks, hoop_tracks)

        # 5. Extract features
        print("  Extracting features...")
        feature_extractor = ShotFeatureExtractor()
        quality_predictor = ShotQualityPredictor()

        video_shots = []

        for frame_num, is_shot in enumerate(shot_frames):
            if is_shot:
                shot_start = max(0, frame_num - 60)
                shot_end = frame_num

                # Get shooter ID
                shooter_id = None
                if frame_num < len(pose_tracks) and pose_tracks[frame_num]:
                    shooter_id = list(pose_tracks[frame_num].keys())[0]

                if shooter_id is not None:
                    # Extract features
                    features = feature_extractor.extract_shot_features(
                        pose_tracks, ball_tracks, hoop_tracks, player_tracks,
                        shot_start, shot_end, shooter_id,
                        frame_width, frame_height
                    )

                    # Predict quality
                    prediction = quality_predictor.predict_shot_quality(features, frame_width)

                    # Store
                    video_shots.append({
                        'frame': frame_num,
                        'prediction': prediction,
                        'actual_result': shot_results[frame_num],
                        'dataset_label': shot_type,  # From folder structure
                        'features': {
                            'biomechanics': features['biomechanics'],
                            'trajectory': features['trajectory'],
                            'contextual': features['contextual']
                        }
                    })

                    print(f"    Shot {len(video_shots)}: Quality={prediction['quality_score']:.1f}, "
                          f"Prob={prediction['success_probability']:.1%}, "
                          f"Result={shot_results[frame_num]}")

        print(f"  ✓ Found {len(video_shots)} shots")
        return video_shots

    except Exception as e:
        print(f"  ✗ Error processing video: {e}")
        return None

def process_basketball51_dataset(dataset_root, output_dir, model_path):
    """
    Process entire Basketball-51 dataset.

    Args:
        dataset_root: Root directory of Basketball-51 dataset
        output_dir: Directory to save extracted features
        model_path: Path to YOLO model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Expected shot types (folders)
    shot_types = ['2pt', '3pt', 'free_throw', 'mid_range']

    all_shots = []
    video_count = 0

    print("=" * 70)
    print("BASKETBALL-51 DATASET PROCESSING")
    print("=" * 70)

    for shot_type in shot_types:
        shot_type_dir = os.path.join(dataset_root, shot_type)

        if not os.path.exists(shot_type_dir):
            print(f"\n⚠️  Directory not found: {shot_type_dir}")
            continue

        print(f"\n{'=' * 70}")
        print(f"Processing {shot_type.upper()} shots")
        print(f"{'=' * 70}")

        # Find all video files
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI']:
            video_files.extend(glob.glob(os.path.join(shot_type_dir, ext)))

        print(f"Found {len(video_files)} videos")

        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] {os.path.basename(video_path)}")

            shots = process_single_video(video_path, output_dir, model_path, shot_type)

            if shots:
                for shot in shots:
                    shot['video_file'] = os.path.basename(video_path)
                    shot['video_path'] = video_path
                    all_shots.append(shot)

            video_count += 1

    # Save aggregated results
    output_file = os.path.join(output_dir, 'basketball51_features.json')
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': 'Basketball-51',
            'total_videos': video_count,
            'total_shots': len(all_shots),
            'shots': all_shots
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total videos processed: {video_count}")
    print(f"Total shots extracted: {len(all_shots)}")
    print(f"Features saved to: {output_file}")

    # Generate summary statistics
    generate_summary(all_shots, output_dir)

    return all_shots

def generate_summary(shots, output_dir):
    """Generate summary statistics of extracted features."""
    import numpy as np

    summary = {
        'shot_types': {},
        'quality_distribution': {
            'excellent': 0,
            'good': 0,
            'average': 0,
            'poor': 0
        },
        'average_scores': {
            'quality': [],
            'biomechanics': [],
            'trajectory': [],
            'contextual': []
        }
    }

    for shot in shots:
        # Count by shot type
        shot_type = shot.get('dataset_label', 'unknown')
        if shot_type not in summary['shot_types']:
            summary['shot_types'][shot_type] = 0
        summary['shot_types'][shot_type] += 1

        # Quality distribution
        category = shot['prediction']['quality_category']
        summary['quality_distribution'][category] += 1

        # Average scores
        summary['average_scores']['quality'].append(shot['prediction']['quality_score'])
        summary['average_scores']['biomechanics'].append(
            shot['prediction']['component_scores']['biomechanics']
        )
        summary['average_scores']['trajectory'].append(
            shot['prediction']['component_scores']['trajectory']
        )
        summary['average_scores']['contextual'].append(
            shot['prediction']['component_scores']['contextual']
        )

    # Calculate averages
    for key in summary['average_scores']:
        values = summary['average_scores'][key]
        summary['average_scores'][key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    # Save summary
    summary_file = os.path.join(output_dir, 'dataset_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary statistics saved to: {summary_file}")

    # Print summary
    print("\n" + "-" * 70)
    print("DATASET SUMMARY")
    print("-" * 70)
    print(f"Shot Types:")
    for shot_type, count in summary['shot_types'].items():
        print(f"  {shot_type}: {count}")

    print(f"\nQuality Distribution:")
    for quality, count in summary['quality_distribution'].items():
        print(f"  {quality}: {count}")

    print(f"\nAverage Quality Score: {summary['average_scores']['quality']['mean']:.1f}")
    print(f"  Biomechanics: {summary['average_scores']['biomechanics']['mean']:.1f}")
    print(f"  Trajectory: {summary['average_scores']['trajectory']['mean']:.1f}")
    print(f"  Contextual: {summary['average_scores']['contextual']['mean']:.1f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process Basketball-51 dataset')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to Basketball-51 dataset root directory')
    parser.add_argument('--output', type=str, default='basketball51_output',
                       help='Output directory for extracted features')
    parser.add_argument('--model', type=str, default='basketball_predictor_V3.pt',
                       help='Path to YOLO model')

    args = parser.parse_args()

    process_basketball51_dataset(args.dataset, args.output, args.model)
