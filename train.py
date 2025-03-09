import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.models import resnet50

from transformer import DETR, DETRTransformer

# First make sure the DETR model is properly defined
# Let's assume all your DETR classes are already defined as in your code


# Define a proper backbone wrapper to match the interface
class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, return_layers, out_channels=2048):
        super().__init__()
        self.backbone = backbone
        self.return_layers = return_layers
        self.out_channels = out_channels

    def forward(self, x):
        # For simplicity, just return the last layer's output
        for name, module in self.backbone.named_children():
            x = module(x)
            if name == self.return_layers[-1]:
                break
        return x


# Function to build the DETR model
def build_detr(backbone, num_classes=91, num_queries=100):
    """
    Build a complete DETR model given a backbone (ResNet)
    """
    d_model = 256

    transformer = DETRTransformer(
        d_model=d_model,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    )

    model = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=num_queries,
        d_model=d_model,
    )

    # Initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


# Create backbone (ResNet50 with frozen BatchNorm)
resnet_backbone = resnet50(pretrained=True)

# Remove the last two layers (avgpool and fc)
backbone_without_fc = nn.Sequential(*list(resnet_backbone.children())[:-2])

# Create wrapped backbone
backbone = BackboneWithFPN(
    backbone=backbone_without_fc,
    return_layers=["layer4"],
    out_channels=2048,  # ResNet50's last layer channels
)

# Build DETR model
model = build_detr(backbone, num_classes=20, num_queries=100)  # VOC has 20 classes

# Download Pascal VOC dataset
# This will download VOC 2012 by default
train_dataset = VOCDetection(
    root="./data",
    year="2012",
    image_set="train",
    download=True,
    transform=transforms.Compose(
        [
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
)

# Create VOC class to index mapping
VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
class_to_idx = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

# Create a simple dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=2,  # Small batch size for testing
    shuffle=True,
    collate_fn=lambda x: x,  # Prevent automatic batching to process samples individually
)


def process_voc_annotations(sample):
    image, target_dict = sample

    annotations = target_dict["annotation"]["object"]
    # Handle case when there's only one object (not in a list)
    if not isinstance(annotations, list):
        annotations = [annotations]

    boxes = []
    labels = []

    for obj in annotations:
        bbox = obj["bndbox"]
        # Convert to [x_min, y_min, x_max, y_max] format and normalize
        img_width = float(target_dict["annotation"]["size"]["width"])
        img_height = float(target_dict["annotation"]["size"]["height"])

        x_min = float(bbox["xmin"]) / img_width
        y_min = float(bbox["ymin"]) / img_height
        x_max = float(bbox["xmax"]) / img_width
        y_max = float(bbox["ymax"]) / img_height

        # Convert to [center_x, center_y, width, height] format for DETR
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        boxes.append([center_x, center_y, width, height])

        # Get class label
        class_name = obj["name"]
        # Map class name to index
        class_idx = class_to_idx.get(
            class_name, 0
        )  # Default to background if not found
        labels.append(class_idx)

    # If no objects, add a dummy background object
    if len(boxes) == 0:
        boxes.append([0, 0, 0, 0])
        labels.append(0)  # Background class

    return {
        "image": image,
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# Define a simple Hungarian matcher for DETR
def hungarian_matcher(outputs, targets, num_queries=100):
    """
    Very simplified version of Hungarian matcher.
    In a real implementation, this would use the Hungarian algorithm.
    """
    # For simplicity, let's just use a greedy assignment
    pred_boxes = outputs["pred_boxes"]  # shape: [batch_size, num_queries, 4]
    pred_logits = outputs[
        "pred_logits"
    ]  # shape: [batch_size, num_queries, num_classes+1]

    batch_size = pred_boxes.shape[0]
    indices = []

    for i in range(batch_size):
        target_boxes = targets["boxes"][i]  # shape: [num_targets, 4]
        target_labels = targets["labels"][i]  # shape: [num_targets]

        num_targets = target_boxes.shape[0]
        if num_targets == 0:
            indices.append(([], []))
            continue

        # Compute cost between predictions and targets
        # L1 distance for boxes
        cost_bbox = torch.cdist(pred_boxes[i], target_boxes, p=1)

        # Class probability cost (negative log likelihood for the right class)
        class_probs = pred_logits[i].softmax(-1)  # [num_queries, num_classes+1]
        cost_class = -class_probs[:, target_labels]

        # Combined cost
        cost = cost_bbox + cost_class.T

        # Greedy assignment (not optimal, but simple)
        pred_indices = []
        tgt_indices = []

        for j in range(min(num_targets, num_queries)):
            # Find minimum cost assignment
            min_idx = torch.argmin(cost)
            tgt_idx = min_idx // cost.shape[1]
            pred_idx = min_idx % cost.shape[1]

            # Add to indices
            pred_indices.append(pred_idx.item())
            tgt_indices.append(tgt_idx.item())

            # Set the cost of used elements to inf
            cost[tgt_idx, :] = float("inf")
            cost[:, pred_idx] = float("inf")

        indices.append((pred_indices, tgt_indices))

    return indices


# Define DETR loss function
def detr_loss(outputs, targets, matcher):
    """
    Simplified DETR loss.
    """
    pred_logits = outputs["pred_logits"]  # [batch_size, num_queries, num_classes+1]
    pred_boxes = outputs["pred_boxes"]  # [batch_size, num_queries, 4]

    batch_size = pred_logits.shape[0]
    num_queries = pred_logits.shape[1]

    # Hungarian matching to find optimal assignment
    indices = matcher(outputs, targets)

    # Classification loss
    cls_loss = 0
    # Box loss
    l1_loss = 0

    for i in range(batch_size):
        pred_idx, tgt_idx = indices[i]

        if len(pred_idx) == 0:  # No objects
            continue

        # Classification loss for matched pairs
        matched_logits = pred_logits[i, pred_idx]
        matched_labels = targets["labels"][i, tgt_idx]
        cls_loss += nn.CrossEntropyLoss()(matched_logits, matched_labels)

        # Box L1 loss for matched pairs
        matched_boxes_pred = pred_boxes[i, pred_idx]
        matched_boxes_tgt = targets["boxes"][i, tgt_idx]
        l1_loss += nn.L1Loss()(matched_boxes_pred, matched_boxes_tgt)

    # Normalize by number of objects
    total_objects = sum(len(idx[0]) for idx in indices)
    if total_objects > 0:
        cls_loss /= total_objects
        l1_loss /= total_objects

    # Add no-object classification loss
    no_object_loss = 0
    for i in range(batch_size):
        pred_idx, _ = indices[i]

        # All queries that didn't match to any object should predict "no object" (class 0)
        no_object_idx = [j for j in range(num_queries) if j not in pred_idx]
        no_object_logits = pred_logits[i, no_object_idx]
        no_object_target = torch.zeros(
            len(no_object_idx), dtype=torch.long, device=pred_logits.device
        )

        if len(no_object_idx) > 0:
            no_object_loss += nn.CrossEntropyLoss()(no_object_logits, no_object_target)

    if batch_size > 0:
        no_object_loss /= batch_size

    # Total loss
    total_loss = cls_loss + 5 * l1_loss + no_object_loss

    return total_loss, {
        "cls_loss": cls_loss.item() if total_objects > 0 else 0,
        "l1_loss": l1_loss.item() if total_objects > 0 else 0,
        "no_object_loss": no_object_loss.item() if batch_size > 0 else 0,
    }


# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_idx, samples in enumerate(train_loader):
        optimizer.zero_grad()

        # Process each sample in the batch
        batch_images = []
        batch_boxes = []
        batch_labels = []

        for sample in samples:
            processed = process_voc_annotations(sample)
            batch_images.append(processed["image"])
            batch_boxes.append(processed["boxes"])
            batch_labels.append(processed["labels"])

        # Stack images into a batch
        images = torch.stack(batch_images).to(device)

        # Create targets dict
        targets = {"boxes": batch_boxes, "labels": batch_labels}

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss, loss_dict = detr_loss(outputs, targets, hungarian_matcher)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            print(
                f"  Class Loss: {loss_dict['cls_loss']:.4f}, Box Loss: {loss_dict['l1_loss']:.4f}, No-Object Loss: {loss_dict['no_object_loss']:.4f}"
            )

    print(
        f"Epoch {epoch} completed, Average Loss: {epoch_loss / len(train_loader):.4f}"
    )

# Save the trained model
torch.save(model.state_dict(), "detr_voc.pth")
