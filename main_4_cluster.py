# ================ Imports and Data loading ===================== #
import os

# Neues Arbeitsverzeichnis festlegen
os.chdir('/Users/nicolasvogel/Dokumente/16_ZHAW_MSc/V5_6_NeuralNetworks&DeepLearning/NNDL_Project_Repo')

# Überprüfen, ob es geklappt hat
print("Aktuelles Arbeitsverzeichnis:", os.getcwd())


from torchvision import transforms
from torchvision.datasets import ImageFolder

# choose directory of pictures
data_dir = "./images_original"

# Load full the dataset with original image size
# Define transformer:
transform=transforms.Compose([transforms.Resize((288, 432)), transforms.ToTensor()])

# load data
dataset = ImageFolder(data_dir,  transform=transform)

# check output
# print(dataset)

# ================ Data preprocessing ======================= #
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from torch.utils.data import Subset, DataLoader

#  Extracting Tags
targets = [dataset[i][1] for i in range(len(dataset))]  # assuming (x, label)

# Stratified Split: 90% Train+Val, 10% Test
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_val_idx, test_idx in sss.split(X=targets, y=targets):
    pass

# # Debug: Testverteilung
# test_targets = [targets[i] for i in test_idx]
# print("Test:", Counter(test_targets))

# Stratified K-Fold on Train+Val (90%)
train_val_targets = [targets[i] for i in train_val_idx]
skf = StratifiedKFold(n_splits=5)
batch_size = 32

for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_idx, train_val_targets)):
    # Map indices back to original dataset
    train_indices = [train_val_idx[i] for i in train_idx]
    val_indices = [train_val_idx[i] for i in val_idx]

    train_targets = [targets[i] for i in train_indices]
    val_targets = [targets[i] for i in val_indices]

    # print(f"\nFold {fold+1}")
    # print("Train:", Counter(train_targets))
    # print("Validation:", Counter(val_targets))

    # Erstelle Subsets und Loader
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)


# print(f"Number of training samples: {len(train_subset)}")
# print(f"Number of validation samples: {len(val_subset)}")
# print(f"Number of test samples: {len(test_subset)}")



# ================ Model building ======================= #
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 18 * 27, 1024)  # adjusted to your input size (288x432 / 4)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (3x288x432) -> (16x144x216)
        x = self.pool(F.relu(self.conv2(x)))  # (16x144x216) -> (32x72x108)
        x = self.pool2(F.relu(self.conv3(x)))  # (32x72x108) -> (64x36x54)
        x = self.pool2(F.relu(self.conv4(x)))  # (64x36x54) -> (64x18x27)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 64 x H/2 x W/2

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 128 x H/4 x W/4

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 256 x H/8 x W/8

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 512 x H/16 x W/16

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 512 x H/32 x W/32
        )

        # Angepasst für Inputgröße 288x432: Output = 512 x 9 x 13
        self.classifier = nn.Sequential(
            nn.Linear(512 * 9 * 13, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ================ Model and hyperparam initializing ======================= #

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model
num_classes = len(dataset.classes)

# use simple model or VGG19
use_VGG19 = False

if use_VGG19:
    model = VGG19(num_classes).to(device)
else:
    model = SimpleCNN(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

# Count Layers
num_Conv_layers = str(model).count("Conv")
num_linear_layers = str(model).count("Linear")


# ================ Model training ======================= #
import time
import pandas as pd
from datetime import date

training_logs = []

# Training loop
num_epochs = 1

for epoch in range(num_epochs):
    start_time = time.time()

    # --- Training ---
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss
    train_acc = correct / total

    # --- Evaluation ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    end_time = time.time()
    epoch_time = end_time - start_time

    # --- Logging ---
    training_logs.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "epoch_time_sec": epoch_time
    })

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f} | "
          f"Time: {epoch_time:.2f}s")

# # convert lr to string for filename
# lr_str = str(lr)
# lr_str = lr_str.split(".")
# lr_str = lr_str[1]

# Optional: Convert to DataFrame and save
save_model_data = True
if save_model_data:
    logs_df = pd.DataFrame(training_logs)
    filename_logs = f"{date.today().strftime('%y%m%d')}_Train-log_conv{num_Conv_layers}_lin{num_linear_layers}_ep{num_epochs}_lr{lr}.csv"
    logs_df.to_csv(filename_logs, index=True)
    print("Training logs saved to file")


# ================ Plot training process ======================= #
import matplotlib.pyplot as plt

# initi plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Accuracy subplot ---
axes[0].plot(logs_df["epoch"], logs_df["train_accuracy"], label="Train Accuracy", marker='o')
axes[0].plot(logs_df["epoch"], logs_df["val_accuracy"], label="Validation Accuracy", marker='x')
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(False)

# --- Loss subplot ---
axes[1].plot(logs_df["epoch"], logs_df["train_loss"], label="Train Loss", marker='o')
axes[1].plot(logs_df["epoch"], logs_df["val_loss"], label="Validation Loss", marker='x')
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(False)

plt.tight_layout()

# Optional: Save Plot to file
save_plot_files = True
if save_plot_files:
    filename_acc_loss = f"{date.today().strftime('%y%m%d')}_Acc-Loss_conv{num_Conv_layers}_lin{num_linear_layers}_ep{num_epochs}_lr{lr}.png"
    plt.savefig(filename_acc_loss, dpi=600)
    print("Progression Plots saved to file")
plt.show()


# ================ Plot clf & cm of test data set ======================= #
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

# Save predictions
y_true = []
y_pred = []


def save_predictions(model, test_loader):
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return y_true, y_pred

save_predictions(model, test_loader)

# Extract labels and classification report
label_idx = [dataset.class_to_idx[i] for i in dataset.classes]
clf_report = classification_report(y_true, y_pred, labels=label_idx, target_names=dataset.classes, output_dict=True)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=label_idx)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Classification Report Heatmap
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, ax=ax1)
ax1.set_title('Classification Report')

# Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
disp.plot(ax=ax2, colorbar=False)  # Avoid duplicate colorbars
ax2.set_title('Confusion Matrix (Test set)')
ax2.grid(False)

plt.tight_layout()

# Optional save plot to file
save_plot_files = True
if save_plot_files:
    filename_clf_cm = f"{date.today().strftime('%y%m%d')}_clf-cm_conv{num_Conv_layers}_lin{num_linear_layers}_ep{num_epochs}_lr{lr}.png"
    plt.savefig(filename_clf_cm, dpi=600)
    print("Analytical Plots saved to file")


plt.show()


# ================ Save Model summary to text file ======================= #
def save_model_summary(model, filepath="model_summary.txt", epochs=None, lr=None, optimizer=None, batch_size=None,
                       device=None, max_per_class=None, epoch_times=None):
    with open(filepath, "w") as f:
        f.write("--- Model Architecture ---\n")
        f.write(str(model))
        f.write("\n\n--- Hyperparameters ---\n")
        if epochs is not None:
            f.write(f"Epochs: {epochs}\n")
        if lr is not None:
            f.write(f"Learning Rate: {lr}\n")
        if optimizer is not None:
            f.write(f"Optimizer: {optimizer.__class__.__name__}\n")
        if batch_size is not None:
            f.write(f"Batch Size: {batch_size}\n")
        if device is not None:
            f.write(f"Device: {device}\n")
        if max_per_class is not None:
            f.write(f"Subset size: {max_per_class}\n")

        if epoch_times is not None:
            total_time = sum(epoch_times)
            f.write(f"\n--- Runtime ---\n")
            f.write(f"Total training time: {total_time:.2f} seconds\n")
            f.write(f"Mean training time: {total_time / num_epochs:.2f} seconds/epoch\n")

        f.write("\n--- Parameters ---\n")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Trainable parameters: {trainable_params}\n")


filename_model_summary = f"{date.today().strftime('%y%m%d')}_Model-Summary_conv{num_Conv_layers}_lin{num_linear_layers}_ep{num_epochs}_lr{lr}.txt"

epoch_times = logs_df['epoch_time_sec']


save_summary_txt = True
if save_summary_txt:
    save_model_summary(model,
                       filepath=filename_model_summary,
                       epochs=num_epochs,
                       lr=lr,
                       optimizer=optimizer,
                       batch_size=batch_size,
                       device=device,
                       max_per_class=None,
                       epoch_times=epoch_times)
    print("Model Summary saved to file")
