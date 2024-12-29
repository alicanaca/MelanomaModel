import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from wtfml.data_loaders.image import ClassificationLoader
import albumentations
from albumentations.pytorch import ToTensorV2

class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(SEResnext50_32x4d, self).__init__()
        self.base_model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=None)
        if pretrained is not None:
            self.base_model.load_state_dict(
                torch.load("C:/Users/alica/Desktop/MelanomaModel/se_resnext50_32x4d-a260b3a4.pth")
            )
        self.l0 = nn.Linear(2048, 1)
    
    def forward(self, image):
        batch_size, _, _, _ = image.shape
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        out = self.l0(x)
        return out

def get_test_loader(test_data_path, test_csv, batch_size=16):
    df = pd.read_csv(test_csv)
    images = df.image_name.values.tolist()
    images = [os.path.join(test_data_path, i + ".jpg") for i in images]
    targets = df.target.values

    aug = albumentations.Compose([
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), 
                                 std=(0.229, 0.224, 0.225), 
                                 max_pixel_value=255.0, always_apply=True),
        ToTensorV2()
    ])

    test_dataset = ClassificationLoader(
        image_paths=images,
        targets=targets,
        resize=None,
        augmentations=aug,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return test_loader, targets

def evaluate_model(model_path, test_loader, device):
    model = SEResnext50_32x4d(pretrained=None)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    predictions = []
    targets = []
    losses = []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1).type_as(outputs))
            losses.append(loss.item())

            preds = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(preds)
            targets.extend(labels.cpu().numpy())
    
    avg_loss = np.mean(losses)
    predictions = np.array(predictions).ravel()
    auc = metrics.roc_auc_score(targets, predictions)
    return avg_loss, auc, predictions, targets

def plot_metrics(avg_loss, auc, predictions, targets):
    plt.figure(figsize=(12, 5))

    # Plot ROC Curve
    fpr, tpr, _ = metrics.roc_curve(targets, predictions)
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color="blue", label=f"AUC = {auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.bar(["Validation Loss"], [avg_loss], color="orange")
    plt.title("Validation Loss")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data_path = "C:/archive/test"
    test_csv = "C:/archive/test.csv"
    model_path = "model_fold_0.bin"

    # Get test loader
    test_loader, targets = get_test_loader(test_data_path, test_csv)

    # Evaluate model
    avg_loss, auc, predictions, targets = evaluate_model(model_path, test_loader, device)

    # Plot metrics
    plot_metrics(avg_loss, auc, predictions, targets)
