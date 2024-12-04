from collections import deque
import cv2
import numpy as np
from mediapipe.python.solutions.pose import Pose
from ultralytics import YOLO
from PipedProcess import PipedProcess
import os
import pandas as pd
import time

class CameraProcess(PipedProcess):
    def __init__(self, camera_id: int, yolo_model_path: str, mediapipe_params: dict):
        super().__init__()
        self.camera_id = camera_id
        self.yolo_model_path = yolo_model_path
        self.mediapipe_params = mediapipe_params
        self.frame_queue = deque(maxlen=150)  # 최근 150개의 프레임
        self.keypoints_queue = deque(maxlen=150)  # 최근 150개의 키포인트
        self.last_valid_keypoints = np.zeros(99)  # 초기값: 0으로 채움
        self.counter = 0
        self.is_fall_detected = False
        self.post_fall_frames = 0  # 낙상 이후 추가 프레임 카운트


    def init(self):
        self.camera = cv2.VideoCapture(self.camera_id)
        self.yolo_model = YOLO(self.yolo_model_path)
        self.pose_model = Pose(**self.mediapipe_params)


    def process(self, input_data: None):

        # 1. 카메라에서 프레임 읽기
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        cv2.imshow('Camera Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt("User exited.")
            
        # 2. 프레임 저장
        self.frame_queue.append(frame)
        person_detected = False
        keypoints = np.zeros(99)  # 기본값: 키포인트 0

        # YOLO로 사람 감지 및 ROI 추출
        results = self.yolo_model.predict(frame, stream=True, classes=[0], verbose=False)
        for r in results:
            for box in r.boxes:
                person_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_img = frame[y1:y2, x1:x2]

                # MediaPipe로 키포인트 추출
                rgb_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                pose_results = self.pose_model.process(rgb_img)

                if pose_results.pose_landmarks:
                    keypoints = self.extract_keypoints(pose_results)
                    if self.detect_fall(pose_results):  # 낙상 조건 확인
                        self.counter += 1
                        if self.counter == 5:  # 낙상 감지
                            self.is_fall_detected = True
                            #print("fall detection!")
                            self.counter = 0

                            
                            return self.collect_frames_for_lstm()

        # 사람이 감지되지 않거나 키포인트 추출 실패 시 기본값 유지
        self.keypoints_queue.append(keypoints)

        frames = list(self.frame_queue)
        keypoints = list(self.keypoints_queue)
        #print(f"Collected {len(frames)} frames and {len(keypoints)} keypoints")
        return None
    
    def collect_frames_for_lstm(self):
        frames = list(self.frame_queue)
        keypoints = list(self.keypoints_queue)

        #print(f"Collected {len(frames)} frames and {len(keypoints)} keypoints for LSTM.")

        return frames, keypoints



    @staticmethod
    def extract_keypoints(results):
        # 33개의 관절 좌표 추출
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z])
        return np.array(keypoints).flatten()  # 1D 배열 (99개 값)


    def detect_fall(self, results):
        if not results.pose_landmarks:  # pose_landmarks가 None인 경우 바로 반환
            return False

        # 주요 랜드마크 추출
        left_hip = results.pose_landmarks.landmark[23]
        right_hip = results.pose_landmarks.landmark[24]
        left_foot = results.pose_landmarks.landmark[31]
        right_foot = results.pose_landmarks.landmark[32]
        nose = results.pose_landmarks.landmark[0]

        # 엉덩이와 발의 평균 y좌표
        hip_avg_y = (left_hip.y + right_hip.y) / 2
        foot_avg_y = (left_foot.y + right_foot.y) / 2

        # 낙상의 기준
        # 1. 엉덩이가 발보다 아래에 위치
        hip_below_foot = hip_avg_y > (foot_avg_y - 0.2)  # 기존보다 조금 완화

        # 2. 머리(코)가 엉덩이보다 낮은 경우 추가
        head_below_hip = nose.y > hip_avg_y + 0.05  # 조금 더 완화된 기준

        # 3. 두 발의 간격이 너무 좁아지는 현상을 완화
        foot_distance = abs(left_foot.x - right_foot.x)
        feet_too_close = foot_distance < 0.10  # 기존보다 넉넉한 기준

        # 조건 중 두 개 이상을 만족하면 낙상으로 판단
        conditions_met = sum([hip_below_foot, head_below_hip, feet_too_close])
        return conditions_met >= 2

        