from ultralytics import YOLO
import supervision as sv

class HoopTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            # Hoops are static, so we can use a lower confidence if needed
            batch_detections = self.model.predict(batch_frames, conf=0.3) 
            detections += batch_detections
        return detections

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
        tracks = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # WE USE "hoop" HERE BASED ON YOUR INPUT
            hoop_class_id = cls_names_inv.get("Hoop", None)
            
            tracks.append({})
            
            # If the model didn't find the 'hoop' class in its config, skip
            if hoop_class_id is None:
                continue

            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            chosen_bbox = None
            max_confidence = 0
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]

                if cls_id == hoop_class_id:
                    # Pick the highest confidence hoop if multiple are found
                    if max_confidence < confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            if chosen_bbox is not None:
                # We store it under ID 1 for simplicity
                tracks[frame_num][1] = {"bbox": chosen_bbox}

        return tracks