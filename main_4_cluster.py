from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Subset
import collections
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from datetime import date
import matplotlib.pyplot as plt

#==============  Data Loading and preprocessing ============== #
# Data directory
data_dir = "./images_original"

# Load full the dataset with original image size
dataset = ImageFolder(data_dir, transform=transforms.Compose([transforms.Resize((288, 432)), # Adjust to original image size  ---------------------  WHY??
transforms.ToTensor()]))

# Load the dataset with resized images
# dataset = ImageFolder(data_dir, transform=transforms.Compose([
#     transforms.Resize((150, 150)),
#     transforms.ToTensor()]))

# Load a subset of the dataset*
# criterion for subset, and  initialization
class_counts = collections.defaultdict(int)
max_per_class = 99  # maximum number of spectrograms per genre -------------------------------------- define subset size HERE
genre_indices = []

# Iterate through the dataset and select indices of spectrogram's*
for idx, (img, label) in enumerate(dataset):
    if class_counts[label] < max_per_class:
        genre_indices.append(idx)
        class_counts[label] += 1
    # stopping criteria if all genres contain 10 spectrogram's
    if len(class_counts) == len(dataset.classes) and all(c >= max_per_class for c in class_counts.values()):
        break

# created subset with the selected indices*
subset = Subset(dataset, genre_indices)


#==============  Train Test split for sub dataset ============== #
# Define the split ratio
train_ratio = 0.8
test_ratio = 0.2



# Collect indices and labels
all_indices = list(range(len(subset)))
all_labels = [subset[i][1] for i in all_indices]  # Integer-Labels

# Stratified Split
train_indices, test_indices = train_test_split(
    all_indices,
    train_size=train_ratio,
    stratify=all_labels,
    random_state=42  # for reproducibility
)

# create subsets
train_subset = Subset(subset, train_indices)
test_subset = Subset(subset, test_indices)

# Create data loaders (load the train and validation into batches)
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)


#==============  Classification Base for Metrics ============== #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        out = self(inputs)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'train_loss': loss, 'train_acc': acc}

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}


    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, time: {:.2f}s".format(
                epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc'],
                result['epoch_time']))

#============== Class for Dynamic model building ============== #
class CNN_dynamic(ImageClassificationBase):
    def __init__(self, conv_configs, linear_sizes):
        """
        conv_configs: list of tuples -> [(out_channels, kernel_size, padding), ...]
        linear_sizes: list with numbers of neurons for each fc layer -> [input_size, hidden1, hidden2, ..., output_size]
        """
        super().__init__()
        self.device = device or torch.device("cpu")
        self.conv_layers = nn.ModuleList()
        input_channels = 3  # the input pictures have only one channel
        for output_channels, kernel_size, padding, use_pool in conv_configs:
            self.conv_layers.append(nn.Conv2d(input_channels, output_channels, kernel_size, padding=padding))
            self.conv_layers.append(nn.ReLU())
            if use_pool:  # functionality to avoid pooling at the end of a layer
                self.conv_layers.append(nn.MaxPool2d(2, 2))
            input_channels = output_channels  # reassign the input channel to the output channels of the previous layer

        self.flatten = nn.Flatten()

        self.linear_layers = nn.ModuleList()
        for i in range(len(linear_sizes) - 1):
            self.linear_layers.append(nn.Linear(linear_sizes[i], linear_sizes[i + 1]))
            if i < len(linear_sizes) - 2:
                self.linear_layers.append(nn.ReLU())

        self.linear_layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        for layer in self.linear_layers:
            x = layer(x)
        return x


#============== Function for Training loop ============== #
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        train_losses = []
        train_accs = []

        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get both loss and accuracy
            result = model.training_step((inputs, labels))
            train_losses.append(result['train_loss'])
            train_accs.append(result['train_acc'])

            # Backprop using just the loss
            result['train_loss'].backward()
            optimizer.step()
            optimizer.zero_grad()

        # Evaluate on validation set
        result = evaluate(model, val_loader)

        # Add average training metrics
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_acc'] = torch.stack(train_accs).mean().item()
        result['epoch_time'] = time.time() - start_time

        # Print metrics
        model.epoch_end(epoch, result)

        history.append(result)

    return history


#============== Model initialization and training ============== #
# Instantiate the model
# initialize conv layers (for test purposes, runtime per epoch on M1 approx. 15 seconds)
conv_configs = [(4, 3, 'same', True), (4, 3, 'same', True),
                            (8, 3, 'same', True), (8, 3, 'same', True)]
linear_sizes = [3888, 1024, 128, 10]

# initialize actual model
# conv_configs_VGG = [(64, 3, 'same', False), (64, 3, 'same', True),
#                                      (128, 3, 'same', False), (128, 3, 'same', True),
#                                      (256, 3, 'same', False), (256, 3, 'same', False), (256, 3, 'same', False), (256, 3, 'same', True),
#                                      (512, 3, 'same', False), (512, 3, 'same', False), (512, 3, 'same', False),  (512, 3, 'same', True),
#                                      (512, 3, 'same', False), (512, 3, 'same', False), (512, 3, 'same', False),  (512, 3, 'same', True)]
#
# linear_sizes_VGG = [4096, 1024, 128, 10]   # Layer size  of the first one might be incorrect and has to be adjusted...

model = CNN_dynamic(conv_configs, linear_sizes).to(device)

# Number of epochs
num_epochs = 7

# Optimizer
opt_func = torch.optim.Adam

# Learning rate
lr = 0.001

# fitting the model on training data and record the result after each epoch
history = fit(num_epochs, lr, model, train_loader, test_loader, opt_func) # in online article they used val_loader (validation) instead of test_loader



#============== Accuracy Plot  ============== #
num_Conv_layers = len(conv_configs)
num_linear_layers = len(linear_sizes)

def plot_accuracies(history):
    """ Plot the history of accuracies"""
    val_accuracies = [x['val_acc'] for x in history]
    train_accuracies = [x['train_acc'] for x in history]
    plt.plot(val_accuracies, '-rx')
    plt.plot(train_accuracies, '-bx')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
    filename_acc = f"{date.today()}_Accuracy_conv{num_Conv_layers}_ep{num_epochs}_max{max_per_class}.png"
    plt.savefig(filename_acc, dpi=600)

plot_accuracies(history)
plt.clf()


#============== Loss Plot  ============== #
def plot_losses(history):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    filename_loss = f"{date.today()}_Loss_conv{num_Conv_layers}_ep{num_epochs}_max{max_per_class}.png"
    plt.savefig(filename_loss, dpi=600)

plot_losses(history)

#============== Modelsummary to txt file  ============== #
def save_model_summary(model, filepath="model_summary.txt", epochs=None, lr=None, optimizer=None, batch_size=None,
                       device=None, max_per_class=None, epochs_times=None):
    with open(filepath, "w") as f:
        f.write("Model Architecture:\n")
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

        f.write("\n--- Parameters ---\n")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Trainable parameters: {trainable_params}\n")


filename_model_summary = f"{date.today()}_Model-Summary_conv{num_Conv_layers}_ep{num_epochs}_max{max_per_class}.txt"
epoch_times = [entry['epoch_time'] for entry in history]
optimizer = opt_func

save_model_summary(model,
                   filepath=filename_model_summary,
                   epochs=10,
                   lr=0.001,
                   optimizer=optimizer,
                   batch_size=32,
                   device=device,
                   max_per_class=max_per_class,
                   epochs_times=epoch_times)




