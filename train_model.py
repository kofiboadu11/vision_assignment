"""
Train a neural network model on Basketball-51 extracted features.

This script trains a PyTorch model to predict shot success based on
biomechanics, trajectory, and contextual features.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from features import ShotFeatureExtractor
import pickle

class ShotDataset(Dataset):
    """PyTorch Dataset for basketball shot features."""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class ShotQualityNet(nn.Module):
    """
    Neural network for shot quality prediction.

    Architecture:
    - Input: 24 features (biomechanics + trajectory + contextual)
    - Hidden layers with dropout for regularization
    - Output: Shot success probability (0-1)
    """

    def __init__(self, input_size=24):
        super(ShotQualityNet, self).__init__()

        self.network = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            # Second hidden layer
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),

            # Third hidden layer
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),

            # Output layer
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )

    def forward(self, x):
        return self.network(x)

def load_dataset(features_file):
    """
    Load extracted features from Basketball-51 processing.

    Args:
        features_file: Path to basketball51_features.json

    Returns:
        features: numpy array of shape (n_shots, 24)
        labels: numpy array of shape (n_shots,) with 0/1 labels
        metadata: list of shot metadata
    """
    with open(features_file, 'r') as f:
        data = json.load(f)

    extractor = ShotFeatureExtractor()
    features_list = []
    labels_list = []
    metadata_list = []

    print(f"Loading {len(data['shots'])} shots from dataset...")

    for shot in data['shots']:
        # Convert feature dict to vector
        feature_vector = extractor.features_to_vector(shot['features'])

        # Normalize features
        normalized_features = extractor.normalize_features(feature_vector)

        # Label: 1 if made, 0 if missed
        # You'll need to manually annotate these!
        actual_result = shot.get('actual_result')

        if actual_result == 'made':
            label = 1.0
        elif actual_result == 'missed':
            label = 0.0
        else:
            # Skip shots without ground truth labels
            continue

        features_list.append(normalized_features)
        labels_list.append(label)
        metadata_list.append({
            'video': shot.get('video_file'),
            'frame': shot.get('frame'),
            'shot_type': shot.get('dataset_label'),
            'prediction': shot.get('prediction')
        })

    features = np.array(features_list)
    labels = np.array(labels_list)

    print(f"Loaded {len(features)} labeled shots")
    print(f"  Made: {int(labels.sum())}")
    print(f"  Missed: {int(len(labels) - labels.sum())}")

    return features, labels, metadata_list

def train_model(features_file, output_model_path, epochs=100, batch_size=32):
    """
    Train the shot quality prediction model.

    Args:
        features_file: Path to extracted features JSON
        output_model_path: Where to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    # Load data
    features, labels, metadata = load_dataset(features_file)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\nTraining set: {len(X_train)} shots")
    print(f"Test set: {len(X_test)} shots")

    # Create datasets
    train_dataset = ShotDataset(X_train, y_train)
    test_dataset = ShotDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = ShotQualityNet(input_size=features.shape[1]).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    best_test_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device).unsqueeze(1)

            # Forward pass
            outputs = model(features_batch)
            loss = criterion(outputs, labels_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels_batch).sum().item()
            train_total += labels_batch.size(0)

        # Validation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for features_batch, labels_batch in test_loader:
                features_batch = features_batch.to(device)
                labels_batch = labels_batch.to(device).unsqueeze(1)

                outputs = model(features_batch)
                loss = criterion(outputs, labels_batch)

                test_loss += loss.item()
                predictions = (outputs > 0.5).float()
                test_correct += (predictions == labels_batch).sum().item()
                test_total += labels_batch.size(0)

        # Calculate metrics
        train_acc = 100.0 * train_correct / train_total
        test_acc = 100.0 * test_correct / test_total
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)

        # Learning rate scheduling
        scheduler.step(avg_test_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Test Loss: {avg_test_loss:.4f} Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'feature_names': ShotFeatureExtractor().get_feature_names()
            }, output_model_path)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Model saved to: {output_model_path}")

    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train shot quality prediction model')
    parser.add_argument('--features', type=str, required=True,
                       help='Path to basketball51_features.json')
    parser.add_argument('--output', type=str, default='trained_model.pth',
                       help='Output path for trained model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')

    args = parser.parse_args()

    train_model(args.features, args.output, args.epochs, args.batch_size)
