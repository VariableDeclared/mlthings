from ultralytics import YOLO

# Load a model
model = YOLO("/Users/peterdesousa/projects/ml-experiments/yolo11/runs/detect/train7/weights/last.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(["predict/reg1.jpeg", "predict/reg2.jpeg", "predict/reg3.jpeg"])  # return a list of Results objects

# Process results list
i=1
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
   # result.show()  # display to screen
    result.save(filename=f"result{i}.jpg")  # save to disk
    i += 1
