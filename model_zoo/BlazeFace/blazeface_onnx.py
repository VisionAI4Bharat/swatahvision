import swatahVision as sv
import numpy as np
import cv2

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
MODEL_PATH = "blaze_fixed.onnx"
IMAGE_PATH = "your_image.jpg"

CONF_THRESH = 0.5

# ---------------------------------------------
# Load model
# ---------------------------------------------
model = sv.Model(
    model=MODEL_PATH,
    engine=sv.Engine.ONNX,
    hardware=sv.Hardware.CPU
)

# Access ONNX runtime session
session = model.runtime_engine.session

# ---------------------------------------------
# Load image
# ---------------------------------------------
image = sv.Image.load_from_file(IMAGE_PATH)
img = np.array(image)
H, W = img.shape[:2]

# ---------------------------------------------
# Preprocess for BlazeFace
# ---------------------------------------------
img128 = cv2.resize(img, (128,128))

img_np = img128.astype(np.float32) / 255.0
img_np = np.transpose(img_np, (2,0,1))[None,...]

# ---------------------------------------------
# Run inference (ONLY image input)
# ---------------------------------------------
outs = session.run(
    None,
    {
        "image": img_np
    }
)

# ---------------------------------------------
# Parse outputs
# ---------------------------------------------
boxes = outs[0][0]
print("Detections returned:-", boxes.shape[0])

if boxes.ndim == 1:
    boxes = boxes.reshape(1,16)

scores = outs[1][0] if len(outs) > 1 else np.ones(len(boxes))

# Handle no detections
if boxes.shape[0] == 0:
    print("No faces detected")

# ---------------------------------------------
# Draw detections
# ---------------------------------------------
for det, score in zip(boxes, scores):
    print("Score:", score)
    if score < CONF_THRESH:
        continue

    (
        top_y, top_x, bot_y, bot_x,
        ley_x, ley_y, rey_x, rey_y,
        nose_x, nose_y, mou_x, mou_y,
        lea_x, lea_y, rea_x, rea_y
    ) = det

    x1 = int(top_x * W)
    y1 = int(top_y * H)
    x2 = int(bot_x * W)
    y2 = int(bot_y * H)

    if x2-x1 < 5 or y2-y1 < 5:
        continue

    # Draw bounding box
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)

    # Draw confidence
    cv2.putText(
        img,
        f"{score:.2f}",
        (x1, y1-5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2
    )

    # Landmarks
    landmarks = [
        (ley_x, ley_y),
        (rey_x, rey_y),
        (nose_x, nose_y),
        (mou_x, mou_y),
        (lea_x, lea_y),
        (rea_x, rea_y)
    ]

    for nx, ny in landmarks:
        cx = int(nx * W)
        cy = int(ny * H)
        cv2.circle(img, (cx,cy), 4, (0,0,255), -1)

# ---------------------------------------------
# Show result
# ---------------------------------------------
cv2.imshow("BlazeFace Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()