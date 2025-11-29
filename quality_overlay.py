"""
Simple quality score overlay for video frames.
Displays compact shot quality metrics.
"""

import cv2

class QualityOverlay:
    """Lightweight overlay showing shot quality score."""

    def __init__(self):
        self.current_score = None
        self.current_metrics = None
        self.display_frames = 60  # Show for 2 seconds at 30fps

    def draw(self, video_frames, shot_analyses, shot_frames):
        """
        Draw quality scores on video frames.

        Args:
            video_frames: List of frames
            shot_analyses: List of analysis results (one per shot)
            shot_frames: Boolean list of shot detection frames

        Returns:
            Frames with quality overlay
        """
        output_frames = []
        analysis_idx = 0
        frames_remaining = 0

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # Check for new shot
            if shot_frames[frame_num] and analysis_idx < len(shot_analyses):
                analysis = shot_analyses[analysis_idx]
                if analysis:
                    self.current_score = analysis['quality_score']
                    self.current_metrics = analysis['metrics']
                    frames_remaining = self.display_frames
                    analysis_idx += 1

            # Draw overlay if active
            if frames_remaining > 0:
                frame = self._draw_overlay(frame)
                frames_remaining -= 1

                if frames_remaining == 0:
                    self.current_score = None
                    self.current_metrics = None

            output_frames.append(frame)

        return output_frames

    def _draw_overlay(self, frame):
        """Draw compact quality score overlay."""
        if self.current_score is None:
            return frame

        # Position (top right)
        frame_height, frame_width = frame.shape[:2]
        x = frame_width - 280
        y = 20

        # Background box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + 260, y + 140),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Get color based on score
        color = self._get_color(self.current_score)

        # Title
        cv2.putText(frame, "SHOT QUALITY",
                   (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 255), 2)

        # Score
        cv2.putText(frame, f"{self.current_score:.0f}/100",
                   (x + 10, y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                   color, 3)

        # Progress bar
        bar_width = int((self.current_score / 100) * 240)
        cv2.rectangle(frame, (x + 10, y + 75),
                     (x + 250, y + 90),
                     (80, 80, 80), -1)
        cv2.rectangle(frame, (x + 10, y + 75),
                     (x + 10 + bar_width, y + 90),
                     color, -1)

        # Key metrics
        y_offset = y + 110
        metrics_text = self._format_metrics()
        cv2.putText(frame, metrics_text,
                   (x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                   (200, 200, 200), 1)

        return frame

    def _format_metrics(self):
        """Format key metrics for display."""
        if not self.current_metrics:
            return ""

        parts = []

        if self.current_metrics.get('release_angle'):
            parts.append(f"Angle:{self.current_metrics['release_angle']:.0f}°")

        if self.current_metrics.get('arc_height'):
            parts.append(f"Arc:{self.current_metrics['arc_height']:.0f}px")

        if self.current_metrics.get('elbow_angle'):
            parts.append(f"Elbow:{self.current_metrics['elbow_angle']:.0f}°")

        return " | ".join(parts[:3])  # Show max 3 metrics

    def _get_color(self, score):
        """Get color based on score."""
        if score >= 75:
            return (0, 255, 0)      # Green - excellent
        elif score >= 60:
            return (100, 255, 100)  # Light green - good
        elif score >= 45:
            return (0, 255, 255)    # Yellow - average
        else:
            return (0, 100, 255)    # Orange - poor
