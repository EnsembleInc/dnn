import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt  # For data viz
import pandas as pd
import numpy as np
import sys

# removed tqdm usage to avoid notebook-only dependency
# added these for making data loader class more flexible
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

print("System Version:", sys.version)
print("PyTorch version", torch.__version__)
print("Torchvision version", torchvision.__version__)
print("Numpy version", np.__version__)
print("Pandas version", pd.__version__)


# Check for accelerator: prefer CUDA, then Apple MPS (Metal), then CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Apple Metal Performance Shaders (MPS) backend for Apple Silicon
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


class FlexibleImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, csv_file=None, class_map=None):
        """
        Args:
            root_dir (str): Directory with images (or parent folder if using ImageFolder style).
            transform (callable, optional): Transformations to apply to images.
            csv_file (str, optional): Path to CSV file containing 'filename,label'.
            class_map (dict, optional): Mapping {label_name: class_index}. If None, inferred.
        """
        self.root_dir = root_dir
        self.transform = transform

        if csv_file:
            # Case 1: CSV-based dataset
            self.data = pd.read_csv(csv_file)
            self.image_paths = [
                os.path.join(root_dir, f) for f in self.data["filename"]
            ]
            self.labels = self.data["label"].tolist()

            # Build class map (if not provided)
            if class_map is None:
                unique_labels = sorted(set(self.labels))
                self.class_map = {label: idx for idx, label in enumerate(unique_labels)}
            else:
                self.class_map = class_map

            self.targets = [self.class_map[l] for l in self.labels]

        else:
            # Case 2: ImageFolder style
            from torchvision.datasets import ImageFolder

            dataset = ImageFolder(root_dir, transform=transform)
            self.image_paths = [path for path, _ in dataset.samples]
            self.targets = [label for _, label in dataset.samples]
            self.class_map = dataset.class_to_idx
            self.transform = transform  # override from ImageFolder

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.targets[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def classes(self):
        return list(self.class_map.keys())


# dataset = FlexibleImageDataset()

dataset = FlexibleImageDataset(
    root_dir="/Users/personal/Desktop/butterfly/train",
    csv_file="/Users/personal/Desktop/butterfly/Training_set.csv",
    # transform=transform
)

print(dataset.classes)  # ['cat', 'dog']


print(len(dataset))

image, label = dataset[6000]
print(label)
image

# Get a dictionary associating target values with folder names
data_dir = "/Users/personal/Desktop/dl_image_data/train"
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
print(target_to_class)


transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)


dataset = PlayingCardDataset(data_dir, transform)

image, label = dataset[100]
image.shape

# iterate over dataset
for image, label in dataset:
    break

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for images, labels in dataloader:
    break

images.shape, labels.shape

print(labels)


class SimpleCardClassifer(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifer, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output


model = SimpleCardClassifer(num_classes=53)
print(str(model)[:500])

example_out = model(images)
example_out.shape  # [batch_size, num_classes]

# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


criterion(example_out, labels)
print(example_out.shape, labels.shape)

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

train_folder = "/Users/personal/Desktop/dl_image_data/train/"
valid_folder = "/Users/personal/Desktop/dl_image_data/valid/"
test_folder = "/Users/personal/Desktop/dl_image_data/test/"

train_dataset = PlayingCardDataset(train_folder, transform=transform)
val_dataset = PlayingCardDataset(valid_folder, transform=transform)
test_dataset = PlayingCardDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Simple training loop
num_epochs = 1
train_losses, val_losses = [], []

# Prefer CUDA, then Apple MPS (for Apple Silicon), then CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = SimpleCardClassifer(num_classes=53)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # Move inputs and labels to the device
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    print(
        f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}"
    )

plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.legend()
plt.title("Loss over epochs")
plt.show()


import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# Load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)


# Predict using the model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


# Visualization
def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    # Display image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")

    # Display predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()


# Example usage
test_image = "/Users/personal/Desktop/dl_image_data/test/five of diamonds/2.jpg"
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

original_image, image_tensor = preprocess_image(test_image, transform)
probabilities = predict(model, image_tensor, device)

# Assuming dataset.classes gives the class names
class_names = dataset.classes
visualize_predictions(original_image, probabilities, class_names)

from glob import glob

test_images = glob("../input/cards-image-datasetclassification/test/*/*")
test_examples = np.random.choice(test_images, 10)

for example in test_examples:
    original_image, image_tensor = preprocess_image(example, transform)
    probabilities = predict(model, image_tensor, device)

    # Assuming dataset.classes gives the class names
    class_names = dataset.classes
    visualize_predictions(original_image, probabilities, class_names)
