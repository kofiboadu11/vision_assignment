import numpy as np
import json
import os

class ShotQualityPredictor:
    """
    Deep learning model for predicting shot quality and success probability.

    This is a placeholder implementation that uses a rule-based system.
    In production, this would be replaced with a trained neural network (PyTorch/TensorFlow).

    The model takes normalized features and outputs:
    - Shot quality score (0-100)
    - Success probability (0-1)
    - Quality category ('excellent', 'good', 'average', 'poor')
    """

    def __init__(self, model_path=None):
        """
        Initialize the shot quality predictor.

        Args:
            model_path: Path to pre-trained model weights (optional)
        """
        self.model_path = model_path
        self.model = None

        # Feature weights for rule-based scoring
        # In production, these would be learned weights from a neural network
        self.feature_weights = {
            # Biomechanics weights
            'elbow_angle': 0.12,
            'knee_bend': 0.10,
            'shoulder_alignment': 0.08,
            'release_height': 0.09,
            'body_balance': 0.07,
            'follow_through': 0.11,
            # Trajectory weights
            'arc_height': 0.15,
            'release_angle': 0.10,
            'entry_angle': 0.12,
            'trajectory_smoothness': 0.08,
            # Contextual weights
            'distance_penalty': -0.10,
            'defender_pressure': -0.08
        }

    def load_model(self, model_path):
        """
        Load pre-trained model weights.
        In production, this would load a PyTorch/TensorFlow model.
        """
        if os.path.exists(model_path):
            with open(model_path, 'r') as f:
                self.feature_weights = json.load(f)
            return True
        return False

    def save_model(self, model_path):
        """
        Save model weights.
        """
        with open(model_path, 'w') as f:
            json.dump(self.feature_weights, f, indent=2)

    def score_biomechanics(self, features):
        """
        Score biomechanics features.

        Args:
            features: Dictionary of biomechanics features

        Returns:
            Score from 0-100
        """
        score = 0.0
        max_score = 0.0

        # Elbow angle (ideal: 90 degrees)
        if features.get('elbow_angle'):
            ideal_elbow = 90
            elbow_deviation = abs(features['elbow_angle'] - ideal_elbow)
            elbow_score = max(0, 100 - elbow_deviation)
            score += elbow_score * self.feature_weights['elbow_angle']
            max_score += 100 * self.feature_weights['elbow_angle']

        # Knee bend (ideal: 120-140 degrees)
        knee_scores = []
        if features.get('knee_bend_left'):
            ideal_knee = 130
            deviation = abs(features['knee_bend_left'] - ideal_knee)
            knee_scores.append(max(0, 100 - deviation * 2))

        if features.get('knee_bend_right'):
            ideal_knee = 130
            deviation = abs(features['knee_bend_right'] - ideal_knee)
            knee_scores.append(max(0, 100 - deviation * 2))

        if knee_scores:
            avg_knee_score = np.mean(knee_scores)
            score += avg_knee_score * self.feature_weights['knee_bend']
            max_score += 100 * self.feature_weights['knee_bend']

        # Shoulder alignment (ideal: close to 0 degrees - level)
        if features.get('shoulder_alignment'):
            shoulder_score = max(0, 100 - abs(features['shoulder_alignment']) * 2)
            score += shoulder_score * self.feature_weights['shoulder_alignment']
            max_score += 100 * self.feature_weights['shoulder_alignment']

        # Release height (higher is better, normalized)
        if features.get('release_height'):
            # Assume max useful height is 200 pixels above midpoint
            height_score = min(100, (features['release_height'] / 200) * 100)
            score += height_score * self.feature_weights['release_height']
            max_score += 100 * self.feature_weights['release_height']

        # Body balance (lower deviation is better)
        if features.get('body_balance'):
            # Assume good balance is < 50 pixels deviation
            balance_score = max(0, 100 - (features['body_balance'] / 50) * 100)
            score += balance_score * self.feature_weights['body_balance']
            max_score += 100 * self.feature_weights['body_balance']

        # Follow-through (positive value is good)
        if features.get('follow_through'):
            # Good follow-through is 20-60 pixels
            ft_score = min(100, max(0, (features['follow_through'] / 60) * 100))
            score += ft_score * self.feature_weights['follow_through']
            max_score += 100 * self.feature_weights['follow_through']

        if max_score > 0:
            return (score / max_score) * 100
        return 50.0  # Default neutral score

    def score_trajectory(self, features):
        """
        Score trajectory features.
        """
        score = 0.0
        max_score = 0.0

        # Arc height (higher arc is generally better, 100-300 pixels ideal)
        if features.get('arc_height'):
            ideal_arc = 200
            deviation = abs(features['arc_height'] - ideal_arc)
            arc_score = max(0, 100 - (deviation / 100) * 100)
            score += arc_score * self.feature_weights['arc_height']
            max_score += 100 * self.feature_weights['arc_height']

        # Release angle (ideal: 45-55 degrees)
        if features.get('release_angle'):
            ideal_release = 50
            deviation = abs(features['release_angle'] - ideal_release)
            release_score = max(0, 100 - deviation * 2)
            score += release_score * self.feature_weights['release_angle']
            max_score += 100 * self.feature_weights['release_angle']

        # Entry angle (ideal: 45-60 degrees - steeper is better)
        if features.get('entry_angle'):
            ideal_entry = 50
            deviation = abs(features['entry_angle'] - ideal_entry)
            entry_score = max(0, 100 - deviation * 2)
            score += entry_score * self.feature_weights['entry_angle']
            max_score += 100 * self.feature_weights['entry_angle']

        # Trajectory smoothness (lower variance is better)
        if features.get('trajectory_smoothness') is not None:
            # Lower smoothness value is better
            smoothness_score = max(0, 100 - features['trajectory_smoothness'] * 1000)
            score += smoothness_score * self.feature_weights['trajectory_smoothness']
            max_score += 100 * self.feature_weights['trajectory_smoothness']

        if max_score > 0:
            return (score / max_score) * 100
        return 50.0

    def score_contextual(self, features, frame_width):
        """
        Score contextual features and apply penalties.
        """
        score = 100.0

        # Distance penalty (farther = harder)
        if features.get('distance_to_basket'):
            normalized_dist = features['distance_to_basket'] / frame_width
            distance_penalty = normalized_dist * abs(self.feature_weights['distance_penalty']) * 100
            score -= distance_penalty

        # Defender pressure penalty
        if features.get('defender_pressure'):
            pressure_penalty = features['defender_pressure'] * abs(self.feature_weights['defender_pressure']) * 100
            score -= pressure_penalty

        return max(0, min(100, score))

    def predict_shot_quality(self, features, frame_width=1280):
        """
        Predict overall shot quality from extracted features.

        Args:
            features: Dictionary containing all feature groups
            frame_width: Video frame width for normalization

        Returns:
            Dictionary with:
            - quality_score: Overall score 0-100
            - success_probability: Probability of making the shot 0-1
            - quality_category: 'excellent', 'good', 'average', 'poor'
            - component_scores: Breakdown by feature group
        """
        # Score each component
        bio_score = self.score_biomechanics(features.get('biomechanics', {}))
        traj_score = self.score_trajectory(features.get('trajectory', {}))
        ctx_score = self.score_contextual(features.get('contextual', {}), frame_width)

        # Weighted combination
        quality_score = (
            bio_score * 0.35 +   # Biomechanics: 35%
            traj_score * 0.40 +  # Trajectory: 40%
            ctx_score * 0.25     # Context: 25%
        )

        # Convert quality score to success probability
        # Using a sigmoid-like function
        success_probability = 1 / (1 + np.exp(-0.08 * (quality_score - 50)))

        # Categorize quality
        if quality_score >= 80:
            category = 'excellent'
        elif quality_score >= 65:
            category = 'good'
        elif quality_score >= 45:
            category = 'average'
        else:
            category = 'poor'

        return {
            'quality_score': round(quality_score, 2),
            'success_probability': round(success_probability, 3),
            'quality_category': category,
            'component_scores': {
                'biomechanics': round(bio_score, 2),
                'trajectory': round(traj_score, 2),
                'contextual': round(ctx_score, 2)
            }
        }

    def predict_batch(self, feature_list, frame_width=1280):
        """
        Predict shot quality for multiple shots.

        Args:
            feature_list: List of feature dictionaries
            frame_width: Video frame width

        Returns:
            List of prediction dictionaries
        """
        predictions = []
        for features in feature_list:
            prediction = self.predict_shot_quality(features, frame_width)
            predictions.append(prediction)

        return predictions
