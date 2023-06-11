from ultralytics import YOLO
from ultralytics.yolo.v8.pose.predict import PosePredictor
import time
import torch

started_at = time.time()

# Load a model
model = YOLO("yolov8n-pose.pt")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using ", device, " as device")


def on_predict_batch_end(predictor: PosePredictor):
    print(predictor)


# model.add_callback("on_predict_batch_end", on_predict_batch_end)

results = model.track(
    source="./output.mp4",  # file, directory, or glob pattern
    # stream=True,
    device=device,
    imgsz=1920,
    tracker="./bot-sort.config.yaml",
    save=True,  # save prediction
    save_txt=True,  # save results to *.txt
    verbose=False,  # ログを抑制
    # boxes=False,  # save image with boxes
    # show_labels=False,  # save labels to *.txt
)

ended_at = time.time()

print("実行時間: ", ended_at - started_at, "秒")
