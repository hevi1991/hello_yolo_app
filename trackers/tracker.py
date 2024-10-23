import pickle
from ultralytics import YOLO
import supervision as sv
import os
import cv2
import numpy as np
import pandas as pd
from utils import get_width, get_center
import math


class Tracker:
    """追踪器"""

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        dectections = []

        for i in range(0, len(frames), batch_size):
            # 每20个元素一段进行预测
            dectections_batch = self.model.predict(frames[i : i + batch_size], conf=0.1)
            dectections += dectections_batch

        return dectections

    def get_object_tracks(self, frames, read_db=False, db_path=None):

        if read_db and db_path is not None and os.path.exists(db_path):
            with open(db_path, "rb") as file:
                tracker_datas = pickle.load(file)
            return tracker_datas

        detections = self.detect_frames(frames)
        # 数据结果
        tracker_datas = {"players": [], "referees": [], "ball": []}

        # 遍历预测结果
        for frame_index, detection in enumerate(detections):
            # 使用supervision的Detections类将预测结果转换为supervision的检测结果
            sv_detection = sv.Detections.from_ultralytics(detection)
            object_names = detection.names
            object_names_inverse = {value: key for key, value in object_names.items()}

            # 将预测结果中的goalkeeper类别转换为player类别
            for object_index, class_id in enumerate(sv_detection.class_id):
                if object_names[class_id] == "goalkeeper":
                    sv_detection.class_id[object_index] = object_names_inverse["player"]

            # 将修改完的检测结果转换为supervision的检测结果， 生成tacker_id。【可能会丢失帧数】
            get_tracks = self.tracker.update_with_detections(sv_detection)

            # 初始化该帧数据
            tracker_datas["players"].append({})
            tracker_datas["referees"].append({})
            tracker_datas["ball"].append({})

            # 直接使用get_tracks反而会掉帧
            for frame_track in sv_detection:
                # 下标解释： (xyxy, mask, confidence, class_id, tracker_id, data)` for each detection.
                box = frame_track[0].tolist()
                class_id = frame_track[3]
                tracker_id = frame_track[4]

                if class_id == object_names_inverse["player"]:
                    # 跟踪数据解释： [球员数据][所在帧][跟踪目标id] = 目标bounding box坐标 xyxy
                    tracker_datas["players"][frame_index][tracker_id] = {"box": box}
                elif class_id == object_names_inverse["referee"]:
                    tracker_datas["referees"][frame_index][tracker_id] = {"box": box}
                elif class_id == object_names_inverse["ball"]:
                    tracker_datas["ball"][frame_index][1] = {"box": box}

        if db_path is not None:
            with open(db_path, "wb") as file:
                pickle.dump(tracker_datas, file)

        return tracker_datas

    def interpolate_and_bfill_ball_tracks(self, ball_tracks):
        ball_positions = [x.get(1, {}).get("box", []) for x in ball_tracks]
        # 格式转换
        df_ball_tracks = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        # 估测补充方法
        df_ball_tracks = df_ball_tracks.interpolate()
        df_ball_tracks = df_ball_tracks.bfill()
        # 格式转换
        ball_positions = [
            {1: {"box": box_value}} for box_value in df_ball_tracks.to_numpy().tolist()
        ]
        return ball_positions

    def add_annotations(self, frames, track_datas, alpha=0.3):
        # 输出
        output_frames = []

        for frame_index, frame in enumerate(frames):

            frame = frame.copy()
            # 遮罩
            overlay = frame.copy()

            players = track_datas["players"][frame_index]
            referees = track_datas["referees"][frame_index]
            ball = track_datas["ball"][frame_index]

            for track_id, player in players.items():
                color = player.get("team_color", (255, 255, 255))
                frame = self.draw_ellipse(frame, player["box"], color, track_id)
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["box"], (0, 0, 255))

            for _, referee in referees.items():
                frame = self.draw_ellipse(frame, referee["box"], (0, 255, 255))

            for _, bal in ball.items():
                frame = self.draw_triangle(frame, bal["box"], (245, 12, 80))

            # 加边框
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            output_frames.append(frame)

        return output_frames

    def draw_triangle(self, frame, box, rgb_color, alpha=0.3):
        if math.isnan(box[1]):
            return frame

        y = int(box[1])
        x, _ = get_center(box)

        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])

        frame = cv2.drawContours(frame, [triangle_points], -1, rgb_color, cv2.FILLED)
        # 边框
        frame = cv2.drawContours(frame, [triangle_points], -1, (0, 0, 0), 2)

        overlay = frame.copy()

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def draw_ellipse(self, frame, box, rgb_color, track_id=None):

        x_center, _ = get_center(box)
        y_center = int(box[3])

        width = get_width(box)

        cv2.ellipse(
            frame,
            # 椭圆中心坐标
            center=(x_center, y_center),
            # 长短轴
            axes=(width, int(0.35 * width)),
            angle=0.0,
            startAngle=-45.0,
            endAngle=245.0,
            color=rgb_color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        if track_id is not None:

            rect_width = 36
            rect_height = 16
            rect_x1 = x_center - rect_width // 2
            rect_x2 = x_center + rect_width // 2
            rect_y1 = y_center + rect_height // 2
            rect_y2 = rect_y1 + rect_height

            cv2.rectangle(
                frame, (rect_x1, rect_y1), (rect_x2, rect_y2), rgb_color, cv2.FILLED
            )

            text_x1 = rect_x1 + 5
            text_y1 = rect_y1 + rect_height // 2 + 3
            if 9 < track_id < 100:
                text_x1 += rect_width // 4 - 5
            if track_id < 10:
                text_x1 += rect_width // 2 - 8

            cv2.putText(
                frame,
                str(track_id),
                (text_x1, text_y1),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (11, 45, 100),
                2,
            )

        return frame
