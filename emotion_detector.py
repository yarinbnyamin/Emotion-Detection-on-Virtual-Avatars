import cv2
import numpy as np
import torch
from PIL import Image
import mss
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_map = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprise",
    }

    # Load the fine-tuned model
    model_id = "trpakov/vit-face-expression"
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(model_id).to(device)
    model.eval()

    # Load YOLOv11
    model_name = "yolov11n-face.pt"  # https://github.com/akanametov/yolo-face?tab=readme-ov-file#models
    detector = YOLO(model_name)

    # Define screen capture region
    mon = {"top": 100, "left": 0, "width": 1000, "height": 700}

    with mss.mss() as sct:
        while True:
            # Capture the screen
            screen_img = np.array(sct.grab(mon))

            cv2.imwrite("screen_img.jpg", screen_img)
            faces_detected = detector(cv2.imread("screen_img.jpg"))

            for box in faces_detected:
                # Check if any faces were detected
                if len(box.boxes) == 0:
                    break

                x, y, w, h = box.boxes.xywh.int().tolist()[0]
                x -= w // 2
                y -= h // 2
                cv2.rectangle(
                    screen_img,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 0),
                    thickness=2,
                )
                roi_gray = screen_img[
                    y : y + h, x : x + w
                ]  # cropping region of interest i.e. face area from  image

                pil_image = Image.fromarray(roi_gray).convert("RGB")
                inputs = processor(images=pil_image, return_tensors="pt").to(device)

                # Predict the emotion
                with torch.no_grad():
                    outputs = model(**inputs)
                    pred_idx = outputs.logits.argmax(dim=1).item()
                    predicted_emotion = label_map.get(pred_idx, 0)

                text_position = (
                    int(x + w),
                    int(y + h),
                )

                # Display the predicted emotion on the screen
                cv2.putText(
                    screen_img,
                    predicted_emotion,
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Facial emotion analysis ", screen_img)

            if cv2.waitKey(1) & 0xFF == ord("q"):  # Exit when 'q' is pressed
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
