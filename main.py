from assignment import Assigner
import cv2
from ball_assigner import BallAssigner
from trackers.tracker import Tracker
from utils import read_video, save_video


def main():
    video_frames = read_video("videos/football.mp4")

    tracker = Tracker("models/best.pt")

    tracker_datas = tracker.get_object_tracks(
        video_frames, read_db=False, db_path="db/db.pkl"
    )

    tracker_datas["ball"] = tracker.interpolate_and_bfill_ball_tracks(
        tracker_datas["ball"]
    )

    assigner = Assigner()
    assigner.assign_team_color(video_frames[0], tracker_datas["players"][0])

    for frame_index, player_track in enumerate(tracker_datas["players"]):
        for player_id, track in player_track.items():
            team_id = assigner.assign_team_id(
                video_frames[frame_index], track["box"], player_id
            )
            tracker_datas["players"][frame_index][player_id]["team"] = team_id
            tracker_datas["players"][frame_index][player_id]["team_color"] = assigner.team_colors[team_id]

    # Assign Ball Aquisition
    ball_assigner = BallAssigner()
    for frame_index, player_track in enumerate(tracker_datas['players']):
        ball_box = tracker_datas['ball'][frame_index][1]['box']
        assigned_player = ball_assigner.assign_ball_to_player(player_track, ball_box)

        if assigned_player != -1:
            tracker_datas['players'][frame_index][assigned_player]['has_ball'] = True

    output_frames = tracker.add_annotations(video_frames, tracker_datas)
    output_video = save_video(output_frames, "result.mp4")


if __name__ == "__main__":
    main()

    # 遍历第0帧，第一个player
    # for trace_id, player in tracker_datas["players"][0].items():
    #     # 第0帧，第一个player
    #     box = player["box"]
    #     # 第0帧图像
    #     frame = video_frames[0]

    #     # frame切片 y1,y2一维图像高度，x1,x2二维图像宽度
    #     cropped_image = frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
    #     cv2.imwrite("cropped_image.jpg", cropped_image)
    #     break
