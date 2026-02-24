# Pose Estimation
This demo performs human pose estimation using the MoveNet model. It detects 17 body keypoints for each person in an image or video and provides their coordinates along with confidence scores. The detected keypoints can be used for visualization, tracking, or further analysis.

## Install
- clone repository and navigate to example directory

git clone https://github.com/VisionAI4Bharat/swatahVision.git
cd swatahVision/examples/pose_estimation

- setup python environment and activate it [optional]

python3 -m venv venv
source venv/bin/activate

- install required dependencies

pip install -r requirements.txt
