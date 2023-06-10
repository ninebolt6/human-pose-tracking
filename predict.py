from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-pose.pt")

# Use the model
# metrics = model.val()  # evaluate model performance on the validation set

results = model.predict(
    source="./bus.jpg",
    kwargs={
        "size": 640,  # image size
        "save": True,  # save prediction images
        "save_txt": True,  # save results to *.txt
        "boxes": False,  # save image with boxes
        "show_labels": False,  # save labels to *.txt
    },
)  # predict on an image

print(results)
