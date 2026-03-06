# Age & Gender Prediction using SwatahVision (OpenVINO Backend)

This example demonstrates how to perform **Age and Gender prediction**
using the **`age-gender-recognition-retail-0013`** model through the  
**SwatahVision framework**, with **OpenVINO as the inference engine**.

The script loads **face images**, runs inference on **CPU**,  
and prints the predicted **age** and **gender** in the terminal.

---

## 📁 Folder Structure

```bash
age_gender_project/
├── age-gender-recognition-retail-0013.xml
├── age-gender-recognition-retail-0013.bin
├── age_gender_prediction.py
├── age_gender.md
└── assets/
    ├── image_old_man.jpg
    └── image_boy.jpg

```
Add cropped face images inside the assets/ folder.

Example file names:
```bash
assets/image_old_man.jpg

assets/image_boy.jpg
```
✅ Image Guidelines

For best results:
```bash
Each image must contain only one face

Face should be:

Front-facing

Well-lit

Clearly visible

Properly cropped

You may use any face images (male or female).
```
🔧 Requirements
```bash
Python 3.9+

NumPy

OpenCV

SwatahVision
```
🔧 Model Link to download:
```bash
https://huggingface.co/swatah/swatahvision/tree/main/classifiation/age-gender-recognition-retail-0013
```

⚠️ Note: OpenVINO is used internally by SwatahVision.
You do NOT need to write OpenVINO code manually.

🧩 Installation
```bash
1️⃣ Create and Activate Conda Environment (Recommended)
conda create -n swatah_env python=3.9 -y
conda activate swatah_env

2️⃣ Install Required Dependencies
pip install numpy
pip install opencv-python
pip install swatahvision
```

🚀 How to Run

Navigate to your project directory and run:

python age_gender_prediction.py
🧪 Code Explanation
📦 Import Required Libraries
```bash
import swatahvision as sv
import numpy as np
import cv2
🧠 Load Age & Gender Model
model = sv.Model(
    model="age-gender-recognition-retail-0013.xml",
    engine=sv.Engine.OPENVINO,
    hardware=sv.Hardware.CPU
)
Explanation

sv.Engine.OPENVINO → Uses OpenVINO internally

sv.Hardware.CPU → Runs inference on CPU
```
```bash
🖼 Run Inference on Image
image = sv.Image.load_from_file("assets/image_old_man.jpg")
outputs = model(image)[0]
📤 Extract Model Outputs
gender_blob = outputs[0]
age_blob = outputs[1]

gender_blob → Gender probabilities

age_blob → Normalized age value

👨 Process Gender Output
gender_id = int(np.argmax(gender_blob))
gender = "Male" if gender_id == 1 else "Female"

The model returns probabilities in the format:

[Female, Male]
🎂 Process Age Output
age = int(age_blob[0][0][0][0] * 100)
Explanation

The model outputs age as a normalized value (0–1)

It is multiplied by 100 to convert into age in years

⚠️ Age prediction is approximate and may vary slightly.
```

📤 Output

The script prints the predicted age and gender in the terminal:
```bash
Predicted Age   : 62
Predicted Gender: Male

Predicted Age   : 14
Predicted Gender: Male
🧠 Model Information
Property	Value
Model Name	age-gender-recognition-retail-0013
Framework	SwatahVision
Inference Engine	OpenVINO (internal)
Hardware	CPU
Input Size	62 × 62
Outputs	Age & Gender
Outputs Details

Age → Normalized regression value × 100

Gender → Probabilities [Female, Male]
```
⚠️ Notes

```bash
This example expects cropped face images

Only single-face inference is supported

Prediction accuracy depends on:

Face alignment

Lighting conditions

Image quality
```

👨‍💻 Author
```bash
Aarav Agarwal
Atharva Kotkar
MIT Internship – Swatah AI
```