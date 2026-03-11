# Pose Estimation

## Overview

**Pose Estimation** is a computer vision task used to detect and track human body keypoints such as:

- head
- shoulders
- elbows
- wrists
- hips
- knees
- ankles

These keypoints help determine **human posture and movement**.

Common applications include:

- fitness analysis
- sports analytics
- human–computer interaction
- healthcare monitoring
- motion tracking

swatahVision supports pose estimation using models like **MoveNet**.

---

# MoveNet Pose Estimation Example

This example demonstrates how to perform **human pose estimation** using a **MoveNet model** with the **OpenVINO inference engine**.

The script:

1. Loads the MoveNet model
2. Loads an input image
3. Runs inference
4. Extracts pose keypoints
5. Prints the detected keypoints and confidence scores

---

# Code Example

```python
import swatahvision as sv
import numpy as np

# ---------------------------------------------
# Load MoveNet model
# - OpenVINO inference engine
# - Running on CPU
# ---------------------------------------------
model = sv.Model(
    model="movenet.xml",  # Replace with your OpenVINO IR XML file
    engine=sv.Engine.OPENVINO,
    hardware=sv.Hardware.CPU
)

# ---------------------------------------------
# Load input image from file
# ---------------------------------------------
image = sv.Image.load_from_file("assets/sample.jpg")

# ---------------------------------------------
# Run pose estimation inference
# ---------------------------------------------
outs = model(image)

# ---------------------------------------------
# Parse MoveNet output
# - Expected shapes: (1,1,17,3) or (1,17,3)
# - Each keypoint: y, x, confidence
# ---------------------------------------------
output = outs
while isinstance(output, (list, tuple)):
    output = output[0]

output = np.asarray(output)

if output.ndim == 4:
    kps = output[0][0]
elif output.ndim == 3:
    kps = output[0]
else:
    raise ValueError(f"Invalid MoveNet output shape: {output.shape}")

# Convert to (x, y) + confidence
keypoints = np.stack([kps[:, 1], kps[:, 0]], axis=1)
confidence = kps[:, 2]

# ---------------------------------------------
# Print keypoints and confidence
# ---------------------------------------------
for i, (kp, conf) in enumerate(zip(keypoints, confidence)):
    print(f"Keypoint {i}: x={kp[0]:.2f}, y={kp[1]:.2f}, confidence={conf:.2f}")
```

---

# Model

This example uses the **MoveNet pose estimation model**.

MoveNet detects **17 human body keypoints** in an image.

Keypoints include:

| ID | Body Part |
|---|---|
| 0 | Nose |
| 1 | Left Eye |
| 2 | Right Eye |
| 3 | Left Ear |
| 4 | Right Ear |
| 5 | Left Shoulder |
| 6 | Right Shoulder |
| 7 | Left Elbow |
| 8 | Right Elbow |
| 9 | Left Wrist |
| 10 | Right Wrist |
| 11 | Left Hip |
| 12 | Right Hip |
| 13 | Left Knee |
| 14 | Right Knee |
| 15 | Left Ankle |
| 16 | Right Ankle |

---

# Output Format

The MoveNet model returns keypoints in the format:

```
(y, x, confidence)
```

Where:

| Field | Description |
|------|-------------|
| `y` | vertical coordinate |
| `x` | horizontal coordinate |
| `confidence` | confidence score of keypoint |

Example output:

```
Keypoint 0: x=0.52, y=0.18, confidence=0.98
Keypoint 1: x=0.49, y=0.17, confidence=0.96
Keypoint 2: x=0.55, y=0.17, confidence=0.97
```

---

# How It Works

The pipeline consists of the following steps:

### 1 Load Model

The MoveNet model is loaded using the **OpenVINO inference engine**.

```python
model = sv.Model(
    model="movenet.xml",
    engine=sv.Engine.OPENVINO,
    hardware=sv.Hardware.CPU
)
```

---

### 2 Load Image

An image is loaded using the SwatahVision image utility.

```python
image = sv.Image.load_from_file("assets/sample.jpg")
```

---

### 3 Run Inference

The model processes the image and returns pose estimation outputs.

```python
outs = model(image)
```

---

### 4 Parse Model Output

The raw model output is converted into a structured array of:

- keypoints
- confidence scores

---

### 5 Display Results

The script prints detected keypoints with their coordinates and confidence.

---

# Applications

Pose estimation is used in many real-world applications:

- **Fitness tracking**  
- **Sports analysis**  
- **Gesture recognition**  
- **Activity monitoring**  
- **Human motion analysis**

---

# Summary

Pose estimation detects **human body keypoints** and estimates posture from images or videos.

Using **MoveNet with OpenVINO**, swatahVision enables fast and efficient pose detection on CPU hardware.

---

## 👨‍💻 Author

- **Atharva Kotkar**