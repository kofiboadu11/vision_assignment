import cv2
import numpy as np

class ShotQualityDrawer:
    """
    Draws shot quality predictions and feature breakdowns on video frames.
    """

    def __init__(self):
        self.display_frames = 90  # How long to display prediction
        self.current_display_count = 0
        self.current_prediction = None
        self.current_features = None

    def draw(self, video_frames, shot_predictions, shot_frames):
        """
        Draw shot quality predictions on video frames.

        Args:
            video_frames: List of video frames
            shot_predictions: List of prediction dictionaries (one per shot)
            shot_frames: Boolean list indicating shot detection frames

        Returns:
            Video frames with shot quality overlays
        """
        output_frames = []
        prediction_idx = 0

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # Check if a new shot was detected
            if shot_frames[frame_num] and prediction_idx < len(shot_predictions):
                self.current_display_count = self.display_frames
                self.current_prediction = shot_predictions[prediction_idx]
                prediction_idx += 1

            # Draw prediction if active
            if self.current_display_count > 0:
                frame = self._draw_prediction_overlay(frame)
                self.current_display_count -= 1

                # Clear when done
                if self.current_display_count == 0:
                    self.current_prediction = None

            output_frames.append(frame)

        return output_frames

    def _draw_prediction_overlay(self, frame):
        """
        Draw the shot quality prediction overlay on a frame.
        """
        if self.current_prediction is None:
            return frame

        pred = self.current_prediction
        quality_score = pred['quality_score']
        success_prob = pred['success_probability']
        category = pred['quality_category']
        component_scores = pred['component_scores']

        # Position for overlay (right side of frame)
        frame_height, frame_width = frame.shape[:2]
        overlay_x = frame_width - 400
        overlay_y = 50

        # Create semi-transparent overlay
        overlay = frame.copy()

        # Main quality box
        box_height = 280
        cv2.rectangle(overlay, (overlay_x, overlay_y),
                     (overlay_x + 350, overlay_y + box_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y_offset = overlay_y + 30

        # Title
        cv2.putText(frame, "SHOT QUALITY ANALYSIS",
                   (overlay_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_offset += 40

        # Quality Score with color coding
        score_color = self._get_score_color(quality_score)
        cv2.putText(frame, f"Quality Score: {quality_score:.1f}/100",
                   (overlay_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)

        # Draw quality bar
        y_offset += 10
        bar_width = int((quality_score / 100) * 320)
        cv2.rectangle(frame, (overlay_x + 10, y_offset),
                     (overlay_x + 330, y_offset + 20),
                     (100, 100, 100), -1)
        cv2.rectangle(frame, (overlay_x + 10, y_offset),
                     (overlay_x + 10 + bar_width, y_offset + 20),
                     score_color, -1)

        y_offset += 35

        # Success Probability
        prob_color = self._get_score_color(success_prob * 100)
        cv2.putText(frame, f"Success Probability: {success_prob:.1%}",
                   (overlay_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, prob_color, 2)

        y_offset += 30

        # Category
        category_color = {
            'excellent': (0, 255, 0),
            'good': (100, 255, 100),
            'average': (0, 255, 255),
            'poor': (0, 0, 255)
        }.get(category, (255, 255, 255))

        cv2.putText(frame, f"Category: {category.upper()}",
                   (overlay_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, category_color, 2)

        y_offset += 35

        # Component Scores
        cv2.putText(frame, "Component Breakdown:",
                   (overlay_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        y_offset += 25

        components = [
            ('Biomechanics', component_scores['biomechanics']),
            ('Trajectory', component_scores['trajectory']),
            ('Context', component_scores['contextual'])
        ]

        for comp_name, comp_score in components:
            comp_color = self._get_score_color(comp_score)

            # Component name and score
            cv2.putText(frame, f"{comp_name}:",
                       (overlay_x + 15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.putText(frame, f"{comp_score:.1f}",
                       (overlay_x + 200, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, comp_color, 1)

            # Small bar
            bar_width = int((comp_score / 100) * 100)
            cv2.rectangle(frame, (overlay_x + 240, y_offset - 10),
                         (overlay_x + 240 + bar_width, y_offset),
                         comp_color, -1)

            y_offset += 22

        return frame

    def _get_score_color(self, score):
        """
        Get color based on score value.
        Green for high, yellow for medium, red for low.
        """
        if score >= 80:
            return (0, 255, 0)  # Green
        elif score >= 65:
            return (100, 255, 100)  # Light green
        elif score >= 50:
            return (0, 255, 255)  # Yellow
        elif score >= 35:
            return (0, 165, 255)  # Orange
        else:
            return (0, 0, 255)  # Red
