import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from typing import Optional
from torchvision import transforms
from PIL import Image
import time
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(4)
torch.backends.mkldnn.enabled = True
torch.backends.mkldnn.deterministic = False


pro_dir = "/home/bear/AI_GO/pytorch_practice/FacialEmotionRecognition"
project_dir = "/home/bear/AI_GO/pytorch_practice/FacialEmotionRecognition/data"
csv_file = "/home/bear/AI_GO/pytorch_practice/FacialEmotionRecognition/data/labels.csv"
emotion2label = {
    "anger": 0,
    "contempt": 1,
    "disgust": 2,
    "fear": 3,
    "happy": 4,
    "neutral": 5,
    "sad": 6,
    "surprise": 7,
}


class Pic_dataset(Dataset):

    def __init__(self, csv_file, transform, train: Optional[bool] = None):
        super().__init__()
        self.csv_file = pd.read_csv(csv_file)
        self.csv_file["label"] = self.csv_file["label"].map(emotion2label)
        self.train = train
        self.transform = transform
        self.image_paths = []
        self.labels = []

        if self.train:
            self.dataset_dir = os.path.join(project_dir, "Train")
        else:
            self.dataset_dir = os.path.join(project_dir, "Test")

        for type in os.listdir(self.dataset_dir):
            pics_ctype_pth = os.path.join(self.dataset_dir, type)
            for img_name in os.listdir(pics_ctype_pth):
                img_path = os.path.join(pics_ctype_pth, img_name)
                short_img_path = os.path.join(type, img_name)
                result = self.csv_file[self.csv_file["pth"] == short_img_path]
                if not result.empty:
                    label = result["label"].iloc[0]
                else:
                    label = emotion2label.get(type)

                self.image_paths.append(img_path)
                self.labels.append(label)

        print(f"Loaded {len(self.image_paths)} images from {self.dataset_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_pth = self.image_paths[index]
        label = self.labels[index]
        image = self.transform(Image.open(img_pth).convert("RGB"))

        return image, torch.tensor(label, dtype=torch.long)


transform_train = transforms.Compose(
    [
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_dataset = Pic_dataset(
    csv_file=csv_file,
    transform=transform_train,
    train=True,
)

test_dataset = Pic_dataset(
    csv_file=csv_file,
    transform=transform_test,
    train=False,
)


train_dataloader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, num_workers=4, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # Input 3x48x48
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Input 64x24x24
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Input 128x12x12
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Input 256x6x6
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 8),
        )

    def forward(self, x):
        return self.cnn(x)



checkpoint_path = os.path.join(pro_dir, "model_checkpoint.pth")
checkpoint = torch.load(checkpoint_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
learning_rate = 1e-3
epochs = 0
batch_size = 128
last_Loss = None
weight_decay = 1e-4
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=3,
)
writer = SummaryWriter(os.path.join("/home/bear/AI_GO/pytorch_practice/FacialEmotionRecognition", "logs"))

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
scheduler.best = checkpoint["best_loss"]
epochs = checkpoint["epoch"] + 1


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    total_samples = 0
    start_time = time.time()

    for batch, (image, label) in enumerate(dataloader):
        image, label = image.to(device, non_blocking=True), label.to(device, non_blocking=True)
        pred = model(image)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * image.size(0)
        total_samples += image.size(0)

        if batch % 50 == 0:
            batch_time = time.time() - start_time
            samples_per_sec = image.size(0) * 50 / batch_time
            avg_loss = total_loss / total_samples
            print(f"Batch {batch}/{len(dataloader)} | Loss: {avg_loss:.4f} | "
                  f"Samples/sec: {samples_per_sec:.0f} | LR: {optimizer.param_groups[0]['lr']:.1e}")
            start_time = time.time()
    
    return total_loss / total_samples


def test_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device, non_blocking=True), label.to(device, non_blocking=True)

            pred = model(image)
            loss = loss_fn(pred, label)

            total_loss += loss.item() * image.size(0)
            correct += (pred.argmax(1) == label).sum().item()
            total_samples += image.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    
    print(f"Test Results | Accuracy: {100*accuracy:.1f}% | Avg loss: {avg_loss:.4f}")
    return avg_loss, accuracy


try:
    while True:
        print(f"\nEpoch {epochs} {'-'*40}")

        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        writer.add_scalar("Loss/train", train_loss, epochs)
        test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)
        writer.add_scalar("Accuracy/train", test_acc, epochs)
        scheduler.step(test_loss)
        epochs += 1


except KeyboardInterrupt:
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": last_Loss if last_Loss is not None else float("nan"),
            "best_loss": scheduler.best,
        },
        checkpoint_path,
    )

print("Done!")
