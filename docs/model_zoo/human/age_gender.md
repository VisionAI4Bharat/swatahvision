# Age & Gender Prediction – Code Explanation

This document explains how the **Age & Gender prediction script** works using the **swatahVision framework**.

The program loads a **pre-trained AI model**, processes images, and predicts:

- **Age**
- **Gender**

for the face in the image.

---

# What the Program Does

The script performs the following tasks:

1. Loads a pre-trained **Age & Gender model**
2. Loads an image containing a face
3. Runs the AI model on the image
4. Extracts the predicted age and gender
5. Prints the results in the terminal

---

# Libraries Used

| Library | Purpose |
|------|------|
| `swatahVision` | Runs AI models and handles inference |
| `numpy` | Processes numerical data |
| `opencv (cv2)` | Image processing |

---

# Files Used

| File | Purpose |
|----|----|
| `age-gender-recognition-retail-0013.xml` | Model structure |
| `age-gender-recognition-retail-0013.bin` | Model weights |
| `image_old_man.jpg` | Test image 1 |
| `image_boy.jpg` | Test image 2 |

---

# Step-by-Step Explanation

## Step 1 — Import Required Libraries

The program first imports the required libraries.

```python
import swatahvision as sv
import numpy as np
import cv2
```

These libraries allow the program to:

- load AI models
- process images
- handle numerical operations

---

# Step 2 — Define Model and Image Paths

The paths of the model and images are stored in variables.

```python
MODEL_PATH = "age-gender-recognition-retail-0013.xml"
IMAGE_PATH_OLD = "image_old_man.jpg"
IMAGE_PATH_BOY = "image_boy.jpg"
```

| Variable | Description |
|------|------|
| `MODEL_PATH` | Location of the AI model |
| `IMAGE_PATH_OLD` | Image of an older person |
| `IMAGE_PATH_BOY` | Image of a young boy |

---

# Step 3 — Load the Model

The AI model is loaded using the **swatahVision Model class**.

```python
model = sv.Model(
    model=MODEL_PATH,
    engine=sv.Engine.OPENVINO,
    hardware=sv.Hardware.CPU
)
```

### What this means

| Parameter | Description |
|------|------|
| `model` | Path to the model file |
| `engine` | Uses OpenVINO inference engine |
| `hardware` | Runs on CPU |

This prepares the AI model so it can process images.

---

# Step 4 — Load the First Image

The program loads the first image.

```python
img = sv.Image.load_from_file(IMAGE_PATH_OLD)
```

This converts the image file into a format that the model can process.

---

# Step 5 — Run Model Inference

The image is passed to the model.

```python
outputs = model(img)[0]
```

The model analyzes the image and produces predictions.

The output contains:

| Output | Description |
|------|------|
| Gender prediction |
| Age prediction |

---

# Step 6 — Extract Gender Prediction

The gender prediction is stored in a variable called `gender_blob`.

```python
gender_blob = outputs[0]
```

The model returns probabilities for:

| Index | Gender |
|------|------|
| 0 | Female |
| 1 | Male |

The program selects the **highest probability** using:

```python
gender_id = int(np.argmax(gender_blob))
```

Then converts it into readable text:

```python
gender = "Male" if gender_id == 1 else "Female"
```

---

# Step 7 — Extract Age Prediction

The model returns age as a **normalized value between 0 and 1**.

Example:

```
0.45
```

This value is multiplied by **100** to convert it into years.

```python
age = int(age_blob[0][0][0][0] * 100)
```

Example conversion:

| Model Output | Age |
|------|------|
| 0.20 | 20 years |
| 0.45 | 45 years |
| 0.70 | 70 years |

---

# Step 8 — Print the Prediction

The predicted values are displayed in the terminal.

```python
print("Predicted Age   :", age)
print("Predicted Gender:", gender)
```

Example output:

```
Predicted Age   : 63
Predicted Gender: Male
```

---

# Step 9 — Process the Second Image

The same steps are repeated for the second image.

```python
image = sv.Image.load_from_file(IMAGE_PATH_BOY)
outputs = model(image)[0]
```

The program again:

1. extracts gender
2. extracts age
3. prints the results

---

# Overall Workflow

The complete process looks like this:

```
Load Model
      ↓
Load Image
      ↓
Run AI Model
      ↓
Extract Predictions
      ↓
Display Age & Gender
```

---

# What the Model Outputs

| Output | Description |
|------|------|
| Gender Blob | Probability of Male/Female |
| Age Blob | Normalized age prediction |

---

# Important Notes

- The model works best when the image contains **a clear face**.
- The predicted age is an **estimate**, not an exact value.
- Lighting, face angle, and image quality can affect accuracy.

---

# Summary

This script demonstrates how to:

- load a pretrained AI model
- process images using **swatahVision**
- run model inference
- extract age and gender predictions
- display results in the terminal

It is a simple example of **AI-powered face analysis using Python**.

---

# Author

**Aarav Agarwal**