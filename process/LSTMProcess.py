import os
import numpy as np
import cv2
from PipedProcess import PipedProcess
from tensorflow.keras.models import load_model
from datetime import datetime  # 상단에 추가
import time

class LSTMProcess(PipedProcess):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.model = None


    def init(self):
        self.model = load_model(self.model_path)

    def process(self, input_data):
        if input_data is None:
            return None

        try:
            frames, keypoints_sequence = input_data
        except ValueError:
            #print(f"Unexpected input data format: {input_data}")
            return None

        if len(keypoints_sequence) < 150:
            #print("Not enough keypoints for evaluation.")
            return None

        # LSTM 평가 로직
        sequences = np.array(keypoints_sequence)  # (150, 99)
        input_data = np.expand_dims(sequences, axis=0)  # (1, 150, 99)
        results = self.model.predict(input_data)
        if results > 0.7:
            #print("LSTM Evaluation Result:", results)
            #print("Fall detected. Saving video...")
            self.save_video(frames, "fall_detected.mp4")  # 낙상 영상 저장하는 코드!(석운)
            time.sleep(30)
        # else:
        #     print("LSTM Evaluation Result:", results)
        #     print("No fall detected.")
        return results # 이게 0~1값으로 나오는 결과인데 0.7 이상을 낙상으로 감지하게끔 한거야!(석운)
    
    def save_video(self, frames, output_path=None, fps=15):
        """
        프레임을 동영상으로 저장하는 함수.
        """
        if not frames:
            #print("No frames to save as video.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"fall_detected_{timestamp}.mp4"


        # 첫 번째 프레임의 크기 확인
        height, width, _ = frames[0].shape

        # 동영상 파일 초기화
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # .mp4 코덱
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            video.write(frame)  # 프레임 추가

        video.release()
        #print(f"Video saved to {output_path}")

        