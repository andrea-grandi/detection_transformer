import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection

# Download Pascal VOC dataset
# This will download VOC 2012 by default
train_dataset = VOCDetection(
    root="./data",
    year="2012",
    image_set="train",
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
)

# Create a simple dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=2,  # Small batch size for testing
    shuffle=True,
)
