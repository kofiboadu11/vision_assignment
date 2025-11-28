from utils import read_video, save_video, get_center_of_bbox
from trackers import PlayerTracker, BallTracker, HoopTracker # Added HoopTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer, ShotDrawer # Added ShotDrawer

def detect_shots(ball_tracks, hoop_tracks):
    """
    Returns a list of booleans indicating if a shot happened in that frame.
    """
    shot_frames = [False] * len(ball_tracks)

    for frame_num in range(len(ball_tracks)):
        # Get data for this frame
        ball_data = ball_tracks[frame_num].get(1, {})
        hoop_data = hoop_tracks[frame_num].get(1, {})

        if "bbox" not in ball_data or "bbox" not in hoop_data:
            continue

        ball_bbox = ball_data["bbox"]
        hoop_bbox = hoop_data["bbox"]

        # Check collision: Is Ball Center inside Hoop Box?
        ball_center = get_center_of_bbox(ball_bbox)
        
        # hoop_bbox format: [x1, y1, x2, y2]
        x1, y1, x2, y2 = hoop_bbox
        
        # We add a small margin (e.g., 10px) to make it more forgiving
        if (x1 - 10 < ball_center[0] < x2 + 10) and \
           (y1 - 10 < ball_center[1] < y2 + 10):
            shot_frames[frame_num] = True

    return shot_frames

def main():
    # 1. Read Video
    video_path = "input_videos/YTDown.com_YouTube_LeBron-Jokes-After-Steph-Misses-Free-Thr_Media_welHDbZ0KBY_001_720p.mp4"
    video_frames = read_video(video_path)
    
    # 2. Load Trackers
    model_path = "basketball_predictor_V3.pt"
    player_tracker = PlayerTracker(model_path)
    ball_tracker = BallTracker(model_path)
    hoop_tracker = HoopTracker(model_path) # Initialize Hoop Tracker
    
    # 3. Get Tracks
    print("Tracking Players...")
    player_tracks = player_tracker.get_object_tracks(video_frames)
    
    print("Tracking Ball...")
    ball_tracks = ball_tracker.get_object_tracks(video_frames)
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
    
    print("Tracking Hoop...")
    hoop_tracks = hoop_tracker.get_object_tracks(video_frames) # Get Hoop Tracks
    
    # 4. Detect Shots
    print("Detecting Shots...")
    shot_frames = detect_shots(ball_tracks, hoop_tracks)
    
    # 5. Draw Output
    print("Drawing...")
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    shot_drawer = ShotDrawer() # Initialize Shot Drawer
    
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
    
    # Draw shot detection text overlay
    output_video_frames = shot_drawer.draw(output_video_frames, shot_frames)
    
    save_video(output_video_frames, "output_videos/output_video.avi")
    print("Done!")
    
if __name__ == "__main__":
    main()