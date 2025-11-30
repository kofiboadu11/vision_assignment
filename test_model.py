"""
Test trained shot quality model on new videos.
"""
import json
from pathlib import Path
from train_model import ShotQualityModel
from prepare_dataset import Basketball51Dataset


def test_on_video(video_path: str, model_path: str, detector_model_path: str):
    """
    Test trained model on a single video.

    Args:
        video_path: Path to video file
        model_path: Path to trained model
        detector_model_path: Path to YOLO detector model
    """
    # Load trained model
    print(f"Loading trained model from {model_path}...")
    model = ShotQualityModel.load(model_path)

    # Initialize dataset processor for feature extraction
    print("Initializing feature extractor...")
    dataset = Basketball51Dataset(
        dataset_path="",  # Not needed for single video
        model_path=detector_model_path
    )

    # Extract features from video
    print(f"\nProcessing video: {video_path}")
    features = dataset.extract_features_from_video(video_path)

    if features is None:
        print("ERROR: Could not extract features from video")
        return

    # Make prediction
    prediction, probability = model.predict(features)

    # Display results
    print("\n" + "=" * 60)
    print("SHOT QUALITY PREDICTION")
    print("=" * 60)
    print(f"Video: {Path(video_path).name}")
    print(f"\nPrediction: {'MADE' if prediction == 1 else 'MISSED'}")
    print(f"Confidence: {probability:.2%} (probability of making shot)")
    print("\n" + "-" * 60)
    print("EXTRACTED FEATURES:")
    print("-" * 60)

    # Display biomechanics features
    print("\nBiomechanics:")
    bio_features = {
        'elbow_angle': features.get('elbow_angle'),
        'knee_bend': features.get('knee_bend'),
        'release_height_ratio': features.get('release_height_ratio'),
        'shoulder_alignment': features.get('shoulder_alignment'),
        'body_balance': features.get('body_balance'),
    }
    for name, value in bio_features.items():
        if value is not None:
            print(f"  {name:25s}: {value:8.2f}")
        else:
            print(f"  {name:25s}: N/A")

    # Display trajectory features
    print("\nTrajectory:")
    traj_features = {
        'arc_height': features.get('arc_height'),
        'release_angle': features.get('release_angle'),
        'entry_angle': features.get('entry_angle'),
        'shot_distance': features.get('shot_distance'),
        'trajectory_smoothness': features.get('trajectory_smoothness'),
    }
    for name, value in traj_features.items():
        if value is not None:
            print(f"  {name:25s}: {value:8.2f}")
        else:
            print(f"  {name:25s}: N/A")

    # Display scores
    print("\nScores:")
    print(f"  {'biomechanics_score':25s}: {features.get('biomechanics_score', 0):8.2f}")
    print(f"  {'trajectory_score':25s}: {features.get('trajectory_score', 0):8.2f}")

    print("=" * 60)

    return prediction, probability, features


def test_on_dataset(dataset_path: str, model_path: str, output_path: str = None):
    """
    Test trained model on entire dataset and generate detailed report.

    Args:
        dataset_path: Path to dataset features JSON
        model_path: Path to trained model
        output_path: Optional path to save detailed predictions
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = ShotQualityModel.load(model_path)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    print(f"Testing on {len(dataset)} samples...")

    # Make predictions
    results = []
    correct = 0

    for sample in dataset:
        features = sample['features']
        true_label = sample['label']

        prediction, probability = model.predict(features)

        if prediction == true_label:
            correct += 1

        results.append({
            'video_path': sample['video_path'],
            'shot_type': sample['shot_type'],
            'true_label': true_label,
            'predicted_label': prediction,
            'probability': probability,
            'correct': prediction == true_label
        })

    # Calculate overall accuracy
    accuracy = correct / len(dataset)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Total samples:    {len(dataset)}")
    print(f"Correct:          {correct}")
    print(f"Incorrect:        {len(dataset) - correct}")
    print(f"Accuracy:         {accuracy:.2%}")
    print("=" * 60)

    # Break down by shot type
    shot_types = set(r['shot_type'] for r in results)
    for shot_type in sorted(shot_types):
        type_results = [r for r in results if r['shot_type'] == shot_type]
        type_correct = sum(1 for r in type_results if r['correct'])
        type_acc = type_correct / len(type_results) if type_results else 0
        print(f"{shot_type:15s}: {type_acc:.2%} ({type_correct}/{len(type_results)})")

    # Save detailed results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump({
                'overall_accuracy': accuracy,
                'total_samples': len(dataset),
                'correct_predictions': correct,
                'predictions': results
            }, f, indent=2)
        print(f"\nDetailed results saved to: {output_path}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Test single video:  python test_model.py <video_path>")
        print("  Test on dataset:    python test_model.py --dataset <dataset_features.json>")
        sys.exit(1)

    model_path = "shot_quality_model.pkl"
    detector_model_path = "basketball_predictor_V3.pt"

    if sys.argv[1] == "--dataset":
        # Test on dataset
        dataset_path = sys.argv[2] if len(sys.argv) > 2 else "dataset_features.json"
        test_on_dataset(
            dataset_path=dataset_path,
            model_path=model_path,
            output_path="test_predictions.json"
        )
    else:
        # Test on single video
        video_path = sys.argv[1]
        test_on_video(
            video_path=video_path,
            model_path=model_path,
            detector_model_path=detector_model_path
        )
