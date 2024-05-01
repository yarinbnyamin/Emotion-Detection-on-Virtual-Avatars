import os
import cv2
import numpy as np
import torch
import clip
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image
import mss
from anime_face_detector import create_detector


class Net(nn.Module):
    def __init__(self, num_class: int):
        super(Net, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Freeze CLIP parameters
        for param in self.clip.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_class)

    def forward(self, images):
        batch_size = images.size(0)
        image_features = torch.zeros(batch_size, 512).to(self.device)
        for i in range(batch_size):
            image_features[i] = self.clip.encode_image(images[i : i + 1])
        x = self.fc1(image_features)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train_model():
    # Initialize the model
    num_classes = 7  # Number of classes in your dataset
    model = Net(num_classes)
    model.to(model.device)

    # Prepare dataset
    data_dir = "dataset_manga"
    dataset = datasets.ImageFolder(data_dir, transform=model.preprocess)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Define data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(model.device), labels.to(model.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total
        print(
            f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%"
        )

    # Evaluate on validation set and compute confusion matrix
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(model.device), labels.to(model.device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_accuracy = 100.0 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.show()

    # Save the fine-tuned model
    torch.save(model.state_dict(), "fine_tuned_clip_net_manga.pth")


def main():
    # Load the fine-tuned model
    model = Net(7)
    model.load_state_dict(
        torch.load("fine_tuned_clip_net_manga.pth", map_location=torch.device("cpu"))
    )
    model.device = "cpu"
    model.eval()

    detector = create_detector("yolov3", device="cpu")

    # Define screen capture region
    mon = {"top": 100, "left": 0, "width": 1000, "height": 700}

    with mss.mss() as sct:
        while True:
            # Capture the screen
            screen_img = np.array(sct.grab(mon))

            gray_img = cv2.cvtColor(screen_img, cv2.COLOR_BGR2GRAY)

            cv2.imwrite("screen_img.jpg", gray_img)
            faces_detected = detector(cv2.imread("screen_img.jpg"))
            if len(faces_detected) == 0:
                pass
            else:
                faces_detected = tuple(faces_detected[0]["bbox"][:4])
                faces_detected = [
                    (
                        int(faces_detected[0]),
                        int(faces_detected[1]),
                        int(faces_detected[2] - faces_detected[0]),
                        int(faces_detected[3] - faces_detected[1]),
                    )
                ]

            for x, y, w, h in faces_detected:
                cv2.rectangle(
                    screen_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2
                )
                roi_gray = gray_img[
                    y : y + h, x : x + w
                ]  # cropping region of interest i.e. face area from  image

                image = (
                    model.preprocess(Image.fromarray(roi_gray))
                    .unsqueeze(0)
                    .to(model.device)
                )
                predictions = model(image)
                _, max_index = predictions.max(1)

                emotions = (
                    "angry",
                    "crying",
                    "embarrassed",
                    "happy",
                    "pleased",
                    "sad",
                    "shock",
                )
                predicted_emotion = emotions[max_index]

                text_position = (
                    int(x + w / 2),
                    (
                        int(y + h + 30)
                        if y + h + 30 < screen_img.shape[0]
                        else int(y + h - 10)
                    ),
                )

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
    if not os.path.isfile("fine_tuned_clip_net_manga.pth"):
        train_model()
    else:
        main()
