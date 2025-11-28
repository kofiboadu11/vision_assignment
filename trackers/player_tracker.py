from ultralytics import YOLO
import supervision as sv

class PlayerTracker:
    def __init__(self,model_path):
        self.model=YOLO(model_path)
        self.tracker=sv.ByteTrack()
        
    def detect_frames(self,frames):
        batch_size=20
        detections=[]
        for i in range(0,len(frames),batch_size):
            batch_frames=frames[i:i+batch_size]
            batch_detections=self.model.predict(batch_frames,conf=0.5)
            detections+=batch_detections
        return detections
    
    
    def get_object_tracks(self,frames):
        detections=self.detect_frames(frames)
        tracks=[]
        
        for frame_num,detection in enumerate(detections):
            cls_names=detection.names
            cls_names_inv={v:k for k,v in cls_names.items()}

            detection_supervision=sv.Detections.from_ultralytics(detection)
            
            detection_with_tracks=self.tracker.update_with_detections(detection_supervision)
            
            tracks.append({})
            
            for frame_detection in detection_with_tracks:
                bbox=frame_detection[0].tolist()
                cls_id=frame_detection[3]
                track_id=frame_detection[4]
                
                if cls_id== cls_names_inv["Player"]:
                    tracks[frame_num][track_id]={"bbox":bbox}
            
            
        return tracks
        