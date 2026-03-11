import swatahVision as sv
import numpy as np

# ---------------------------------------------
# Configure label annotation
# ---------------------------------------------
label_annotator = sv.LabelAnnotator(
    color=sv.Color.YELLOW,
    text_color=sv.Color.BLACK,
    text_position=sv.Position.TOP_LEFT,
    text_scale=0.7,
    text_padding=8,
    smart_position=False,
)

# ---------------------------------------------
# Configure bounding box annotation
# ---------------------------------------------
box_annotator = sv.BoxAnnotator(sv.Color.YELLOW)

# ---------------------------------------------
# Load BlazeFace ONNX model
# ---------------------------------------------
model = sv.Model(
    model="blazeface.onnx",
    engine=sv.Engine.ONNX,
    hardware=sv.Hardware.CPU
)

# ---------------------------------------------
# Load image
# ---------------------------------------------
image = sv.Image.load_from_file("assets/face.jpg")

# ---------------------------------------------
# Run inference
# ---------------------------------------------
outs = model(image)

# Remove batch dimension
scores = outs[0][0]   # (896,1)
boxes = outs[1][0]    # (896,16)

# Apply sigmoid to scores
scores = 1 / (1 + np.exp(-scores))

img_np = np.array(image)
h, w = img_np.shape[:2]

xyxy = []
conf = []

# ---------------------------------------------
# Convert outputs to bounding boxes
# ---------------------------------------------
for i in range(len(scores)):

    score = float(scores[i][0])

    # lower threshold for BlazeFace
    if score > 0.2:

        y1, x1, y2, x2 = boxes[i][:4]

        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)

        xyxy.append([x1, y1, x2, y2])
        conf.append(score)

# ---------------------------------------------
# Draw detections
# ---------------------------------------------
if len(xyxy) > 0:

    xyxy = np.array(xyxy)
    conf = np.array(conf)

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=conf
    )

    image = label_annotator.annotate(
        scene=image,
        detections=detections
    )

    image = box_annotator.annotate(
        scene=image,
        detections=detections
    )

else:
    print("No faces detected")

# ---------------------------------------------
# Show output
# ---------------------------------------------
try:
    sv.Image.show(image=image)
except:
    pass