from ultralytics import YOLO
from ultralytics.yolo.v8.pose.predict import PosePredictor

# Load a model
model = YOLO("yolov8n-pose.pt")


def on_predict_batch_end(predictor: PosePredictor):
    print(predictor)


model.add_callback("on_predict_batch_end", on_predict_batch_end)

results = model.track(
    source="./output.mp4",  # file, directory, or glob pattern
    stream=True,
    device=0,
    imgsz=1920,
    tracker="./bot-sort.config.yaml",
    save=True,  # save prediction
    save_txt=True,  # save results to *.txt
    # boxes=False,  # save image with boxes
    # show_labels=False,  # save labels to *.txt
)
