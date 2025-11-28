from utils import read_video, save_video, get_center_of_bbox
from trackers import PlayerTracker, BallTracker, HoopTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer, ShotDrawer

def detect_shots(ball_tracks, hoop_tracks):
    """
    Returns two lists:
    - shot_frames: list of booleans indicating if a shot happened in that frame
    - shot_results: list with None, 'made', or 'missed' for each frame
    """
    shot_frames = [False] * len(ball_tracks)
    shot_results = [None] * len(ball_tracks)
    
    frames_to_check_after = 15  # Check ball position for 15 frames after hoop intersection

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
        hoop_center_y = (y1 + y2) / 2
        
        # We add a small margin (e.g., 10px) to make it more forgiving
        if (x1 - 10 < ball_center[0] < x2 + 10) and \
           (y1 - 10 < ball_center[1] < y2 + 10):
            shot_frames[frame_num] = True
            
            # Now determine if it's made or missed by checking future frames
            shot_result = determine_shot_result(
                ball_tracks, 
                frame_num, 
                frames_to_check_after, 
                hoop_center_y
            )
            shot_results[frame_num] = shot_result

    return shot_frames, shot_results


def determine_shot_result(ball_tracks, shot_frame, frames_to_check, hoop_center_y):
    """
    Determine if a shot was made or missed by analyzing ball movement after hoop intersection.
    
    Logic:
    - MADE: Ball moves downward (y increases) after passing through hoop
    - MISSED: Ball bounces back up or moves horizontally (y decreases or stays similar)
    """
    if shot_frame + frames_to_check >= len(ball_tracks):
        return None
    
    # Get ball position at shot detection
    shot_ball_data = ball_tracks[shot_frame].get(1, {})
    if "bbox" not in shot_ball_data:
        return None
    
    shot_ball_center = get_center_of_bbox(shot_ball_data["bbox"])
    shot_y = shot_ball_center[1]
    
    # Track ball's vertical movement in the frames after shot
    y_positions = []
    for i in range(1, frames_to_check + 1):
        future_frame = shot_frame + i
        if future_frame < len(ball_tracks):
            future_ball_data = ball_tracks[future_frame].get(1, {})
            if "bbox" in future_ball_data:
                future_center = get_center_of_bbox(future_ball_data["bbox"])
                y_positions.append(future_center[1])
    
    if len(y_positions) < 5:  # Not enough data
        return None
    
    # Calculate average Y position in the frames after shot
    avg_y_after = sum(y_positions) / len(y_positions)
    
    # Check if ball moved significantly downward (made shot)
    # A made shot should show the ball continuing downward past the hoop
    y_displacement = avg_y_after - shot_y
    
    # If ball moved down by at least 30 pixels on average, it's likely a made shot
    if y_displacement > 30:
        return "made"
    # If ball stayed at similar height or moved up, it's a miss (bounced off rim)
    else:
        return "missed"


def main():
    # 1. Read Video
    video_path = "input_videos/YTDown.com_YouTube_LeBron-Jokes-After-Steph-Misses-Free-Thr_Media_welHDbZ0KBY_001_720p.mp4"
    video_frames = read_video(video_path)
    
    # 2. Load Trackers
    model_path = "basketball_predictor_V3.pt"
    player_tracker = PlayerTracker(model_path)
    ball_tracker = BallTracker(model_path)
    hoop_tracker = HoopTracker(model_path)
    
    # 3. Get Tracks
    print("Tracking Players...")
    player_tracks = player_tracker.get_object_tracks(video_frames)
    
    print("Tracking Ball...")
    ball_tracks = ball_tracker.get_object_tracks(video_frames)
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
    
    print("Tracking Hoop...")
    hoop_tracks = hoop_tracker.get_object_tracks(video_frames)
    
    # 4. Detect Shots and Results
    print("Detecting Shots...")
    shot_frames, shot_results = detect_shots(ball_tracks, hoop_tracks)
    
    # 5. Draw Output
    print("Drawing...")
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    shot_drawer = ShotDrawer()
    
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
    
    # Draw shot detection text overlay AND trajectory (pass shot_results)
    output_video_frames = shot_drawer.draw(output_video_frames, shot_frames, ball_tracks, shot_results)
    
    save_video(output_video_frames, "output_videos/output_video.avi")
    print("Done!")
    
if __name__ == "__main__":
    main()