"""
Train and evaluate shot quality prediction model.
"""
import json
import numpy as np
import pickle
from typing import Dict, List, Tuple
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


class ShotQualityModel:
    """
    Machine learning model for shot quality prediction.
    """

    def __init__(self, model_type: str = 'random_forest'):
        """
        Args:
            model_type: 'random_forest' or 'gradient_boosting'
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.feature_names = None

    def load_dataset(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from JSON file.

        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
        """
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        # Extract features and labels
        X_list = []
        y_list = []

        for sample in dataset:
            features = sample['features']
            label = sample['label']

            # Convert feature dict to array (maintain consistent order)
            if self.feature_names is None:
                self.feature_names = sorted(features.keys())

            feature_vector = [features[name] for name in self.feature_names]
            X_list.append(feature_vector)
            y_list.append(label)

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=int)

        print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Feature names: {self.feature_names}")

        return X, y

    def preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Preprocess features: impute missing values and scale.

        Args:
            X: Feature matrix
            fit: If True, fit the preprocessors. Otherwise use fitted ones.

        Returns:
            Preprocessed feature matrix
        """
        if fit:
            X = self.imputer.fit_transform(X)
            X = self.scaler.fit_transform(X)
        else:
            X = self.imputer.transform(X)
            X = self.scaler.transform(X)

        return X

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model."""
        print("\nTraining model...")
        self.model.fit(X_train, y_train)
        print("Training complete!")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set.

        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating model...")

        # Predictions
        y_pred = self.model.predict(X_test)

        # Get probabilities - handle single class case
        y_pred_proba_full = self.model.predict_proba(X_test)
        if y_pred_proba_full.shape[1] == 2:
            # Both classes present
            y_pred_proba = y_pred_proba_full[:, 1]
        else:
            # Only one class - use its probability
            y_pred_proba = y_pred_proba_full[:, 0]

        # Calculate metrics
        n_classes_test = len(np.unique(y_test))
        n_classes_pred = len(np.unique(y_pred))

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba) if n_classes_test > 1 else 0.0,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'n_classes_test': n_classes_test,
            'n_classes_trained': len(self.model.classes_)
        }

        return metrics

    def print_evaluation(self, metrics: Dict):
        """Print evaluation metrics in a formatted way."""
        print("\n" + "=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)

        # Check for single-class issues
        if metrics.get('n_classes_trained', 2) < 2:
            print("⚠️  WARNING: Model trained on only ONE class!")
            print("   Dataset should include both made and missed shots.")
            print("   Run prepare_dataset.py with balanced data.\n")

        if metrics.get('n_classes_test', 2) < 2:
            print("⚠️  WARNING: Test set contains only ONE class!")
            print("   Metrics may not be meaningful.\n")

        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")

        # Handle confusion matrix for both single and multi-class
        print("\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])

        if cm.shape == (2, 2):
            print(f"              Predicted")
            print(f"              Miss  Made")
            print(f"Actual Miss   {cm[0,0]:4d}  {cm[0,1]:4d}")
            print(f"       Made   {cm[1,0]:4d}  {cm[1,1]:4d}")
        else:
            print(f"  Single class confusion matrix: {cm.tolist()}")

        print("=" * 50)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return {name: imp for name, imp in zip(self.feature_names, importance)}
        return {}

    def save(self, output_path: str):
        """Save trained model to file."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }

        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\nModel saved to: {output_path}")

    @classmethod
    def load(cls, model_path: str):
        """Load trained model from file."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.imputer = model_data['imputer']
        instance.feature_names = model_data['feature_names']

        return instance

    def predict(self, features: Dict) -> Tuple[int, float]:
        """
        Predict shot quality from features.

        Args:
            features: Dictionary of feature values

        Returns:
            (prediction, probability) where prediction is 0 (miss) or 1 (make)
        """
        # Convert to feature vector
        feature_vector = [features.get(name, 0) for name in self.feature_names]
        X = np.array([feature_vector], dtype=float)

        # Preprocess
        X = self.preprocess(X, fit=False)

        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0][1]  # Probability of making shot

        return int(prediction), float(probability)


def train_and_evaluate(dataset_path: str, model_save_path: str, test_size: float = 0.2):
    """
    Complete training and evaluation pipeline.

    Args:
        dataset_path: Path to extracted features JSON
        model_save_path: Path to save trained model
        test_size: Fraction of data to use for testing
    """
    # Initialize model
    model = ShotQualityModel(model_type='random_forest')

    # Load dataset
    X, y = model.load_dataset(dataset_path)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    print(f"Class distribution - Made: {sum(y_train)}/{sum(y_test)}, Missed: {len(y_train)-sum(y_train)}/{len(y_test)-sum(y_test)}")

    # Check for class balance issues
    if len(np.unique(y_train)) < 2:
        print("\n⚠️  ERROR: Training set contains only one class!")
        print("   Cannot train a binary classifier with only one class.")
        print("   Please run: python prepare_dataset.py")
        print("   This will create a balanced dataset with both made and missed shots.")
        return None, None

    if len(np.unique(y_test)) < 2:
        print("\n⚠️  WARNING: Test set contains only one class!")
        print("   Evaluation metrics will be limited.")

    # Preprocess
    X_train = model.preprocess(X_train, fit=True)
    X_test = model.preprocess(X_test, fit=False)

    # Train
    model.train(X_train, y_train)

    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    model.print_evaluation(metrics)

    # Feature importance
    importance = model.get_feature_importance()
    if importance:
        print("\nTop 5 Most Important Features:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, imp in sorted_features[:5]:
            print(f"  {feature:30s}: {imp:.4f}")

    # Save model
    model.save(model_save_path)

    # Save metrics
    metrics_path = model_save_path.replace('.pkl', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    return model, metrics


if __name__ == "__main__":
    # Train model on extracted features
    train_and_evaluate(
        dataset_path="dataset_features.json",
        model_save_path="shot_quality_model.pkl",
        test_size=0.2
    )
