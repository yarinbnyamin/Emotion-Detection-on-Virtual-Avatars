<h1 align="center">Emotion Detection for Virtual Avatars</h1>
<p align="center">
<a href="https://github.com/yarinbnyamin/Emotion-Detection-on-Virtual-Avatars/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
<a href="https://www.python.org/downloads/release/python-31212/"><img alt="Python Version" src="https://img.shields.io/badge/python-3.12-blue"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


# Info

This project implements a real-time emotion recognition pipeline 
for virtual avatar faces using face detection and transformer-based 
facial expression classification.

⚠️ This project is currently under active development. 
Performance is still being optimized and results should be considered preliminary.


# Usage

1. Copy the project and install all requirements:
> pip install -r requirements.txt

2. Download a YOLO face model from:  
> https://github.com/akanametov/yolo-face?tab=readme-ov-file#models

We used `yolov11n-face` (update line 30 in `emotion_detector.py` if you use another model).

3. Run the script:
> python emotion_detector.py 

This will open a window capturing the left half of your screen (customizable in line 34).

4. Place different characters' images in the captured area to detect their emotions in real time.

5. Press `q` to close the program.


# Pipeline

The system is based on the following components:

1. [YOLO-Face](https://github.com/akanametov/yolo-face) – used for real-time face detection.  
2. [ViT-FER](https://huggingface.co/trpakov/vit-face-expression) – used for facial emotion classification.  
3. [UIBVFED: Virtual Facial Expression Dataset](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0231266) – used to evaluate and validate the complete pipeline.

# Citations

If you find our work interesting or the repo useful, please consider citing this [paper](https://arxiv.org/abs/2601.15914):

```
@article{benyamin2026latency,
  title={The Latency Wall: Benchmarking Off-the-Shelf Emotion Recognition for Real-Time Virtual Avatars},
  author={Benyamin, Yarin},
  journal={arXiv preprint arXiv:2601.15914},
  year={2026}
}
```
