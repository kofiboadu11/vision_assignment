from utils import read_video,save_video
from trackers import PlayerTracker,BallTracker
from drawers import(PlayerTracksDrawer,BallTracksDrawer)
def main():
    video_frames=read_video("input_videos/YTDown.com_YouTube_Basketball-Unbelievable-Last-Minute-Winn_Media_Cot6ZGr-pCM_001_720p.mp4")
    
    player_tracker=PlayerTracker("basketball_predictor_V3.pt")
    ball_tracker=BallTracker("basketball_predictor_V3.pt")
    
    player_tracks=player_tracker.get_object_tracks(video_frames)
    
    ball_tracks=ball_tracker.get_object_tracks(video_frames)
    
    ball_tracks=ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks=ball_tracker.interpolate_ball_positions(ball_tracks)
    
    player_tracks_drawer=PlayerTracksDrawer()
    ball_tracks_drawer=BallTracksDrawer()
    
    output_video_frames= player_tracks_drawer.draw(video_frames,player_tracks)
    output_video_frames= ball_tracks_drawer.draw(output_video_frames,ball_tracks)
    
    
    save_video(output_video_frames,"output_videos/output_video.avi")
    
if __name__ == "__main__":
    main()