<h1 align="center">Emotion Detection for Virtual Avatars</h2>
<p align="center">
<a href="https://github.com/yarinbnyamin/Emotion-Detection-on-Virtual-Avatars/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


# Info

Note: This project is in progress, detection is not optimal.

This project aims to help you detect the emotions of different virtual avatars.

# Usage

1. Copy the project and pip install all the requirements:
> pip install -r requirements.txt
2. Download the [Dataset](https://www.kaggle.com/datasets/mertkkl/manga-facial-expressions) and place it in the project dir with the name "dataset_manga"
3. Start the emotion_detector.py file, this will train your model
4. When this is done, start the emotion_detector.py file again
5. This time pop-up screen will appear and capture the left half of your screen for faces (you can customize the detection size and location in line 142)
6. Now you can place different characters' images and detect the emotion of them
7. Pressing the q button will close the program
8. (optional) You can replace the expression name with the assossiated emotion


# Pipeline

I am using:
1. [CLIP](https://github.com/openai/CLIP) library for obtain embeddings for images
2. [Anime Face Detector](https://github.com/hysts/anime-face-detector) library to detect virtual avatars faces
3. [Manga Facial Expressions Dataset](https://www.kaggle.com/datasets/mertkkl/manga-facial-expressions) to fine-tuning the model
