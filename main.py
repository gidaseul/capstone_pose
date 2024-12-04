from process.CameraProcess import CameraProcess
from process.LSTMProcess import LSTMProcess
from PipedProcess import Pipeline

if __name__ == "__main__":
    # CameraProcess 초기화
    camera_process = CameraProcess(
        camera_id=0,
        yolo_model_path="yolo11n.pt",
        mediapipe_params={'min_detection_confidence': 0.5, 'min_tracking_confidence': 0.5}
    )

    # LSTMProcess 초기화
    lstm_process = LSTMProcess(
        model_path="lstm_fall_detection_model.h5"
    )

    # Pipeline 연결
    pipeline = Pipeline(camera_process, lstm_process)
    pipeline.start()
