# Train mode ResNet18 + CIFAR10 <訓練模型>
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader

# ===== 路徑設定 =====
DATA_ROOT = "D:/Anaconda/PythonCode/data"
SAVE_PATH = "D:/Anaconda/PythonCode/data/ckpt_best.pth"

# ===== device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def build_resnet18_cifar10(num_classes=10):
    m = resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m

# ===== CIFAR-10 transform =====
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2023, 0.1994, 0.2010)

train_tfm = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

test_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

trainset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    transform=train_tfm,
    download=False
)

testset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=False,
    transform=test_tfm,
    download=False
)

trainloader = DataLoader(
    trainset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

testloader = DataLoader(
    testset,
    batch_size=256,
    shuffle=False,
    num_workers=0
)

# ===== model / loss / optimizer =====
model = build_resnet18_cifar10(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[60, 80],
    gamma=0.2
)

# ===== evaluation =====
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total

# ===== training =====
best_acc = 0.0
epochs = 150  

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for x, y in trainloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    acc = evaluate(model, testloader)

    print(f"Epoch [{epoch+1:03d}/{epochs}] "
          f"Loss: {running_loss/len(trainloader):.4f} "
          f"Test Acc: {acc*100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save({
            "model": model.state_dict(),
            "acc": best_acc,
            "epoch": epoch + 1
        }, SAVE_PATH)

        print(f"Saved best checkpoint: {SAVE_PATH}, acc={best_acc*100:.2f}%")

print("Training finished.")
print(f"Best Acc: {best_acc*100:.2f}%")
