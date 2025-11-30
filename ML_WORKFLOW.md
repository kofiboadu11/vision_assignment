# Shot Quality Prediction - Machine Learning Workflow

This document explains the complete machine learning pipeline for training a shot quality prediction model on the Basketball-51 dataset.

## Overview

The ML pipeline consists of 3 main steps:

1. **Dataset Preparation**: Extract features from Basketball-51 videos
2. **Model Training**: Train a classifier and evaluate with metrics (accuracy, AUC, etc.)
3. **Model Testing**: Use the trained model to predict shot quality on new videos

## Dataset Structure

Basketball-51 dataset organizes videos by shot type and outcome:

```
Basketball_51 dataset/
├── 3p0/   # Missed 3-point shots (label = 0)
├── 3p1/   # Made 3-point shots (label = 1)
├── 2p0/   # Missed 2-point shots (label = 0)
├── 2p1/   # Made 2-point shots (label = 1)
├── ft0/   # Missed free throws (label = 0)
├── ft1/   # Made free throws (label = 1)
├── mp0/   # Missed mid-range shots (label = 0)
└── mp1/   # Made mid-range shots (label = 1)
```

The digit (0 or 1) indicates whether the shot was **missed (0)** or **made (1)**.

## Step 1: Dataset Preparation

Extract features from all Basketball-51 videos:

```bash
python prepare_dataset.py
```

This script:
- Processes all videos in the Basketball-51 dataset
- Runs object detection and tracking (ball, hoop, player)
- Performs pose estimation on the shooter
- Extracts 16 numerical features per shot:
  - **Biomechanics (5)**: elbow_angle, knee_bend, release_height_ratio, shoulder_alignment, body_balance
  - **Trajectory (5)**: arc_height, release_angle, entry_angle, shot_distance, trajectory_smoothness
  - **Scores (2)**: biomechanics_score, trajectory_score
  - **Context (4)**: is_3point, is_2point, is_freethrow, is_midrange
- Labels each shot as 0 (miss) or 1 (make) based on folder structure
- Saves extracted features to `dataset_features.json`

**Output**: `dataset_features.json` containing all extracted features and labels

**Note**: For initial testing, the script processes the first 10 videos. Edit `max_videos` parameter to process more.

## Step 2: Model Training & Evaluation

Train a Random Forest classifier on the extracted features:

```bash
python train_model.py
```

This script:
- Loads features from `dataset_features.json`
- Splits data into train (80%) and test (20%) sets
- Handles missing values with mean imputation
- Standardizes features with StandardScaler
- Trains a Random Forest classifier with class balancing
- Evaluates on test set with multiple metrics:
  - **Accuracy**: Overall correctness
  - **Precision**: Of predicted makes, how many were actually made
  - **Recall**: Of actual makes, how many were predicted
  - **F1 Score**: Harmonic mean of precision and recall
  - **AUC-ROC**: Area under ROC curve (discrimination ability)
  - **Confusion Matrix**: Breakdown of predictions
- Displays feature importance rankings
- Saves trained model to `shot_quality_model.pkl`
- Saves metrics to `shot_quality_model_metrics.json`

**Output Files**:
- `shot_quality_model.pkl`: Trained model (ready for inference)
- `shot_quality_model_metrics.json`: Evaluation metrics

**Expected Performance** (will vary based on dataset size):
```
Accuracy:  0.75-0.85
Precision: 0.70-0.80
Recall:    0.75-0.85
F1 Score:  0.72-0.82
AUC-ROC:   0.80-0.90
```

## Step 3: Model Testing

### Test on a Single Video

```bash
python test_model.py path/to/video.mp4
```

This will:
- Load the trained model
- Extract features from the video
- Predict whether the shot will be made/missed
- Display confidence score and all extracted features

**Example Output**:
```
============================================================
SHOT QUALITY PREDICTION
============================================================
Video: test_shot.mp4

Prediction: MADE
Confidence: 87.3% (probability of making shot)

------------------------------------------------------------
EXTRACTED FEATURES:
------------------------------------------------------------
Biomechanics:
  elbow_angle              :    89.45
  knee_bend                :    42.31
  release_height_ratio     :     1.45
  shoulder_alignment       :     3.21
  body_balance             :    12.50

Trajectory:
  arc_height               :   125.30
  release_angle            :    48.50
  entry_angle              :    45.20
  shot_distance            :   250.40
  trajectory_smoothness    :    85.60

Scores:
  biomechanics_score       :    82.30
  trajectory_score         :    88.50
============================================================
```

### Test on Entire Dataset

```bash
python test_model.py --dataset dataset_features.json
```

This will:
- Test the model on all samples in the dataset
- Calculate overall accuracy
- Break down accuracy by shot type (3pt, 2pt, free throw, mid-range)
- Save detailed predictions to `test_predictions.json`

## Feature Descriptions

### Biomechanics Features (from Pose Estimation)

1. **elbow_angle**: Angle at elbow joint (shoulder-elbow-wrist)
   - Optimal range: 85-95 degrees
   - Measured at moment of release

2. **knee_bend**: Knee flexion angle from vertical
   - Optimal range: 30-60 degrees
   - Indicates power generation from legs

3. **release_height_ratio**: Release point height relative to shoulder
   - Higher ratio = higher release point
   - Optimal: >1.3 (release above shoulder)

4. **shoulder_alignment**: Deviation of shoulders from horizontal
   - Lower is better (balanced posture)
   - Measured in degrees

5. **body_balance**: Horizontal displacement between hips and shoulders
   - Lower is better (stable base)
   - Measured in pixels

### Trajectory Features (from Ball Tracking)

1. **arc_height**: Maximum height of ball above release point
   - Higher arc generally better
   - Measured in pixels

2. **release_angle**: Initial trajectory angle from horizontal
   - Optimal range: 45-55 degrees

3. **entry_angle**: Angle of ball approaching hoop
   - Optimal range: 40-50 degrees
   - Steeper entry = larger target area

4. **shot_distance**: Euclidean distance from shooter to hoop
   - Measured in pixels
   - Contextual factor (3pt > 2pt)

5. **trajectory_smoothness**: How well trajectory fits parabolic curve
   - Higher is better (cleaner arc)
   - Score: 0-100

### Derived Scores

1. **biomechanics_score**: Weighted combination of biomechanics features (0-100)
2. **trajectory_score**: Weighted combination of trajectory features (0-100)

### Context Features (One-Hot Encoded)

- **is_3point**: 1 if 3-point shot, 0 otherwise
- **is_2point**: 1 if 2-point shot, 0 otherwise
- **is_freethrow**: 1 if free throw, 0 otherwise
- **is_midrange**: 1 if mid-range shot, 0 otherwise

## Model Architecture

**Classifier**: Random Forest Ensemble
- **n_estimators**: 100 trees
- **max_depth**: 10 levels
- **min_samples_split**: 5
- **min_samples_leaf**: 2
- **class_weight**: balanced (handles class imbalance)
- **random_state**: 42 (reproducibility)

**Preprocessing**:
- Mean imputation for missing values
- StandardScaler normalization

## Customization

### Process Full Dataset

Edit `prepare_dataset.py`:
```python
dataset.prepare_dataset(
    output_path="dataset_features.json",
    max_videos=None  # Process all videos
)
```

### Try Different Models

Edit `train_model.py`:
```python
model = ShotQualityModel(model_type='gradient_boosting')  # Instead of 'random_forest'
```

### Adjust Train/Test Split

```python
train_and_evaluate(
    dataset_path="dataset_features.json",
    model_save_path="shot_quality_model.pkl",
    test_size=0.3  # Use 30% for testing
)
```

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ Basketball-51 Dataset (Videos)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ prepare_dataset.py                                          │
│ - Object detection & tracking                               │
│ - Pose estimation                                           │
│ - Feature extraction (16 features per shot)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓ dataset_features.json
                       │
┌─────────────────────────────────────────────────────────────┐
│ train_model.py                                              │
│ - Train/test split                                          │
│ - Random Forest training                                    │
│ - Evaluation (Accuracy, AUC, F1, etc.)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓ shot_quality_model.pkl
                       │
┌─────────────────────────────────────────────────────────────┐
│ test_model.py                                               │
│ - Load trained model                                        │
│ - Predict on new videos                                     │
│ - Display results with confidence                           │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

**Issue**: Low accuracy on test set
- **Solution**: Process more videos in dataset preparation
- **Solution**: Try different model hyperparameters
- **Solution**: Add more features or feature engineering

**Issue**: Class imbalance (too many makes or misses)
- **Solution**: Model uses `class_weight='balanced'` to handle this
- **Solution**: Ensure equal representation in dataset

**Issue**: Missing values in features
- **Solution**: Imputer handles this with mean strategy
- **Solution**: Check pose estimation quality if many features are None

## Next Steps

1. **Process full dataset**: Remove `max_videos` limit
2. **Hyperparameter tuning**: Use GridSearchCV for optimal parameters
3. **Deep learning**: Try neural networks for potentially better performance
4. **Cross-validation**: Use k-fold CV for more robust evaluation
5. **Feature engineering**: Create interaction features or polynomial features
6. **Real-time prediction**: Integrate into live video stream
