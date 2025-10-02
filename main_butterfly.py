import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from PIL import Image

# -------------------------
# Device setup
# -------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)


# -------------------------
# Flexible Dataset
# -------------------------
class FlexibleImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, csv_file=None, class_map=None):
        self.root_dir = root_dir
        self.transform = transform

        if csv_file:
            self.data = pd.read_csv(csv_file)
            self.image_paths = [os.path.join(root_dir, f) for f in self.data.iloc[:, 0]]

            if "label" in self.data.columns:
                self.labels = self.data["label"].tolist()

                if class_map is None:
                    unique_labels = sorted(set(self.labels))
                    self.class_map = {
                        label: idx for idx, label in enumerate(unique_labels)
                    }
                else:
                    self.class_map = class_map

                self.targets = [self.class_map[l] for l in self.labels]
            else:
                self.labels = None
                self.class_map = class_map
                self.targets = [-1] * len(self.image_paths)
        else:
            from torchvision.datasets import ImageFolder

            dataset = ImageFolder(root_dir, transform=transform)
            self.image_paths = [path for path, _ in dataset.samples]
            self.targets = [label for _, label in dataset.samples]
            self.class_map = dataset.class_to_idx
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.targets[idx] if self.targets is not None else -1
        return image, label

    @property
    def classes(self):
        return list(self.class_map.keys()) if self.class_map else []


# -------------------------
# Model Definition
# -------------------------
class ButterflyClassifier(nn.Module):
    def __init__(self, num_classes=75):
        super(ButterflyClassifier, self).__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output


# -------------------------
# Data Setup
# -------------------------
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

train_csv = "/Users/personal/Desktop/butterfly/Training_set.csv"
valid_csv = "/Users/personal/Desktop/butterfly/Validation_set.csv"
test_csv = "/Users/personal/Desktop/butterfly/Testing_set.csv"
root_train = "/Users/personal/Desktop/butterfly/train"
root_validation = "/Users/personal/Desktop/butterfly/validation"
root_test = "/Users/personal/Desktop/butterfly/test"

train_dataset = FlexibleImageDataset(
    root_dir=root_train, csv_file=train_csv, transform=transform
)
val_dataset = FlexibleImageDataset(
    root_dir=root_validation,
    csv_file=valid_csv,
    transform=transform,
    class_map=train_dataset.class_map,
)
test_dataset = FlexibleImageDataset(
    root_dir=root_test,
    csv_file=test_csv,
    transform=transform,
    class_map=train_dataset.class_map,
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Number of classes: {len(train_dataset.classes)}")

# -------------------------
# Training
# -------------------------
model = ButterflyClassifier(num_classes=len(train_dataset.classes))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}"
    )

plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.legend()
plt.title("Loss over epochs")
plt.show()

# -------------------------
# Save model with class map
# -------------------------
save_path = "butterfly_model.pth"
torch.save(
    {"model_state_dict": model.state_dict(), "class_map": train_dataset.class_map},
    save_path,
)
print(f"Model saved to {save_path}")

# -------------------------
# Reload for Inference
# -------------------------
checkpoint = torch.load(save_path, map_location=device)
num_classes = len(checkpoint["class_map"])
model = ButterflyClassifier(num_classes=num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

class_map = checkpoint["class_map"]
class_names = list(class_map.keys())


# -------------------------
# Inference Helpers
# -------------------------
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)


def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


def visualize_topk(original_image, probabilities, class_names, k=5):
    topk_idx = np.argsort(probabilities)[::-1][:k]
    topk_probs = probabilities[topk_idx]
    topk_classes = [class_names[i] for i in topk_idx]

    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))
    axarr[0].imshow(original_image)
    axarr[0].axis("off")
    axarr[1].barh(topk_classes[::-1], topk_probs[::-1])
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Top Predictions")
    axarr[1].set_xlim(0, 1)
    plt.tight_layout()
    plt.show()


# -------------------------
# Example Prediction
# -------------------------
test_image = "/Users/personal/Desktop/butterfly/test/Image_1.jpg"
original_image, image_tensor = preprocess_image(test_image, transform)
probabilities = predict(model, image_tensor, device)
visualize_topk(original_image, probabilities, class_names, k=5)


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import torchvision
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# import timm

# import matplotlib.pyplot as plt  # For data viz
# import pandas as pd
# import numpy as np
# import sys

# # removed tqdm usage to avoid notebook-only dependency
# # added these for making data loader class more flexible
# import os
# import pandas as pd
# from torch.utils.data import Dataset
# from PIL import Image

# print("System Version:", sys.version)
# print("PyTorch version", torch.__version__)
# print("Torchvision version", torchvision.__version__)
# print("Numpy version", np.__version__)
# print("Pandas version", pd.__version__)


# # Check for accelerator: prefer CUDA, then Apple MPS (Metal), then CPU
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     # Apple Metal Performance Shaders (MPS) backend for Apple Silicon
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# print("Using device:", device)


# class FlexibleImageDataset(Dataset):
#     def __init__(self, root_dir, transform=None, csv_file=None, class_map=None):
#         self.root_dir = root_dir
#         self.transform = transform

#         if csv_file:
#             self.data = pd.read_csv(csv_file)

#             # Always build image paths
#             self.image_paths = [os.path.join(root_dir, f) for f in self.data.iloc[:, 0]]

#             # If labels exist
#             if "label" in self.data.columns:
#                 self.labels = self.data["label"].tolist()

#                 if class_map is None:
#                     unique_labels = sorted(set(self.labels))
#                     self.class_map = {
#                         label: idx for idx, label in enumerate(unique_labels)
#                     }
#                 else:
#                     self.class_map = class_map

#                 self.targets = [self.class_map[l] for l in self.labels]

#             else:
#                 # No labels (e.g., test set)
#                 self.labels = None
#                 self.class_map = class_map
#                 self.targets = [-1] * len(self.image_paths)

#         else:
#             from torchvision.datasets import ImageFolder

#             dataset = ImageFolder(root_dir, transform=transform)
#             self.image_paths = [path for path, _ in dataset.samples]
#             self.targets = [label for _, label in dataset.samples]
#             self.class_map = dataset.class_to_idx
#             self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)

#         label = self.targets[idx] if self.targets is not None else -1
#         return image, label

#     @property
#     def classes(self):
#         return list(self.class_map.keys()) if self.class_map else []


# dataset = FlexibleImageDataset(
#     root_dir="/Users/personal/Desktop/butterfly/train",
#     csv_file="/Users/personal/Desktop/butterfly/Training_set.csv",
#     # transform=transform
# )

# print(f"Class names: {dataset.classes}")
# print(f"Number of classes: {len(dataset.classes)}")


# print(len(dataset))

# image, label = dataset[6000]
# print(f"{str(image)}, \n, {label}")
# image

# # # Get a dictionary associating target values with folder names
# root_dir = "/Users/personal/Desktop/dl_image_data/train"
# # target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
# # print(target_to_class)


# transform = transforms.Compose(
#     [
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#     ]
# )


# dataset = FlexibleImageDataset(root_dir, transform)

# image, label = dataset[100]
# image.shape

# # iterate over dataset
# for image, label in dataset:
#     break

# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# for images, labels in dataloader:
#     break

# print(f"{images.shape}, \n, {labels.shape}")

# print(f"here you go ---> {labels}, \n, {labels.shape}, dtype: {labels.dtype}")


# class ButterflyClassifier(nn.Module):
#     def __init__(self, num_classes=53):
#         super(ButterflyClassifier, self).__init__()
#         # Where we define all the parts of the model
#         self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
#         self.features = nn.Sequential(*list(self.base_model.children())[:-1])

#         enet_out_size = 1280
#         # Make a classifier
#         self.classifier = nn.Sequential(
#             nn.Flatten(), nn.Linear(enet_out_size, num_classes)
#         )

#     def forward(self, x):
#         # Connect these parts and return the output
#         x = self.features(x)
#         output = self.classifier(x)
#         return output


# model = ButterflyClassifier(num_classes=len(train_dataset.classes))
# model.to(device)

# # Loss function

# criterion = nn.CrossEntropyLoss()

# # Optimizer
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# print(str(model)[:500])

# example_out = model(images)
# example_out.shape  # [batch_size, num_classes]


# criterion(example_out, labels)
# print(example_out.shape, labels.shape)

# transform = transforms.Compose(
#     [
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#     ]
# )


# train_csv = "/Users/personal/Desktop/butterfly/Training_set.csv"
# valid_csv = "/Users/personal/Desktop/butterfly/Validation_set.csv"  # not available
# test_csv = "/Users/personal/Desktop/butterfly/Testing_set.csv"
# root_train = "/Users/personal/Desktop/butterfly/train"
# root_validation = "/Users/personal/Desktop/butterfly/validation"
# root_test = "/Users/personal/Desktop/butterfly/test"

# train_dataset = FlexibleImageDataset(
#     root_dir=root_train, csv_file=train_csv, transform=transform
# )
# val_dataset = FlexibleImageDataset(
#     root_dir=root_validation,
#     csv_file=valid_csv,
#     transform=transform,
#     class_map=train_dataset.class_map,
# )
# test_dataset = FlexibleImageDataset(
#     root_dir=root_test,
#     csv_file=test_csv,
#     transform=transform,
#     class_map=train_dataset.class_map,
# )
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# # Simple training loop
# num_epochs = 1
# train_losses, val_losses = [], []

# # Prefer CUDA, then Apple MPS (for Apple Silicon), then CPU
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# model = ButterflyClassifier(num_classes=53)
# model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(num_epochs):
#     # Training phase
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_loader:
#         # Move inputs and labels to the device
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * labels.size(0)
#     train_loss = running_loss / len(train_loader.dataset)
#     train_losses.append(train_loss)

#     # Validation phase
#     model.eval()
#     running_loss = 0.0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             # Move inputs and labels to the device
#             images, labels = images.to(device), labels.to(device)

#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             running_loss += loss.item() * labels.size(0)
#     val_loss = running_loss / len(val_loader.dataset)
#     val_losses.append(val_loss)
#     print(
#         f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}"
#     )

# plt.plot(train_losses, label="Training loss")
# plt.plot(val_losses, label="Validation loss")
# plt.legend()
# plt.title("Loss over epochs")
# plt.show()


# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np


# # Load and preprocess the image
# def preprocess_image(image_path, transform):
#     image = Image.open(image_path).convert("RGB")
#     return image, transform(image).unsqueeze(0)


# # Predict using the model
# def predict(model, image_tensor, device):
#     model.eval()
#     with torch.no_grad():
#         image_tensor = image_tensor.to(device)
#         outputs = model(image_tensor)
#         probabilities = torch.nn.functional.softmax(outputs, dim=1)
#     return probabilities.cpu().numpy().flatten()


# # Visualization
# def visualize_predictions(original_image, probabilities, class_names):
#     fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

#     # Display image
#     axarr[0].imshow(original_image)
#     axarr[0].axis("off")

#     # Display predictions
#     axarr[1].barh(class_names, probabilities)
#     axarr[1].set_xlabel("Probability")
#     axarr[1].set_title("Class Predictions")
#     axarr[1].set_xlim(0, 1)

#     plt.tight_layout()
#     plt.show()


# # Example usage
# test_image = "/Users/personal/Desktop/butterfly/test/Image_1.jpg"
# transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

# original_image, image_tensor = preprocess_image(test_image, transform)
# probabilities = predict(model, image_tensor, device)

# # Assuming dataset.classes gives the class names
# class_names = dataset.classes
# visualize_predictions(original_image, probabilities, class_names)

# # from glob import glob

# # test_images = glob("../input/cards-image-datasetclassification/test/*/*")
# # test_examples = np.random.choice(test_images, 10)

# # for example in test_examples:
# #     original_image, image_tensor = preprocess_image(example, transform)
# #     probabilities = predict(model, image_tensor, device)

# #     # Assuming dataset.classes gives the class names
# #     class_names = dataset.classes
# #     visualize_predictions(original_image, probabilities, class_names)
