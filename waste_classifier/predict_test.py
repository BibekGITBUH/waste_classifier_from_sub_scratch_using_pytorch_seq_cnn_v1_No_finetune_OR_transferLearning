"""
Test model predictions on a folder of images organized by class.
Assumes folder structure: folder/<class_name>/*.jpg

Calculates accuracy, precision, recall, F1-score, and confusion matrix.

Usage:
    python predict_test.py --folder datasets/split/test/
    python predict_test.py --folder datasets/split/val/ --model custom_cnn
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import config
from train import build_model


def get_model(model_name: str = "custom_cnn"):
    """Load and return the model."""
    device = config.DEVICE
    model = build_model(model_name).to(device)
    ckpt_path = Path(config.CHECKPOINT_DIR) / f"{model_name}_best.pt"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model, device


def predict_image(image_path, model, device):
    """Predict class for a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
    except:
        return None
    
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        pred_idx = outputs.argmax(dim=1).item()
    
    return config.CLASS_NAMES[pred_idx]


def test_on_folder(folder_path: str, model_name: str = "custom_cnn"):
    """
    Test model on a folder organized by class subfolders.
    
    Expected structure:
        folder/
            ├── glass/
            │   ├── image1.jpg
            │   ├── image2.jpg
            │   └── ...
            ├── metal/
            │   ├── image1.jpg
            │   └── ...
            └── ...
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found at {folder_path}")
    
    if not folder_path.is_dir():
        raise NotADirectoryError(f"{folder_path} is not a directory")
    
    # Load model
    model, device = get_model(model_name)
    print(f"✓ Loaded model from: {Path(config.CHECKPOINT_DIR) / f'{model_name}_best.pt'}\n")
    
    # Find all class subfolders
    class_folders = {}
    for cls in config.CLASS_NAMES:
        cls_path = folder_path / cls
        if cls_path.exists() and cls_path.is_dir():
            class_folders[cls] = cls_path
    
    if not class_folders:
        print(f"❌ No class folders found in {folder_path}")
        print(f"Expected folders: {', '.join(config.CLASS_NAMES)}")
        return
    
    print(f"Found {len(class_folders)} class folders\n")
    
    # Collect all images and their true labels
    true_labels = []
    pred_labels = []
    image_paths = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    total_images = 0
    processed_images = 0
    
    # Count total images first
    for cls, cls_path in class_folders.items():
        images = [f for f in cls_path.iterdir() if f.suffix in image_extensions]
        total_images += len(images)
    
    print(f"Processing {total_images} images...\n")
    
    # Process images
    for cls, cls_path in sorted(class_folders.items()):
        images = [f for f in cls_path.iterdir() if f.suffix in image_extensions]
        
        for i, image_path in enumerate(sorted(images)):
            pred = predict_image(image_path, model, device)
            
            if pred is None:
                continue
            
            true_labels.append(cls)
            pred_labels.append(pred)
            image_paths.append(str(image_path))
            processed_images += 1
            
            # Progress indicator
            status = "✓" if pred == cls else "✗"
            print(f"\r[{processed_images}/{total_images}] {status} {cls}: {image_path.name}", end="")
    
    print("\n")
    
    if processed_images == 0:
        print("❌ No valid images processed")
        return
    
    # Convert to numpy arrays
    y_true = np.array([config.CLASS_NAMES.index(label) for label in true_labels])
    y_pred = np.array([config.CLASS_NAMES.index(label) for label in pred_labels])
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(config.NUM_CLASSES), zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # Print results
    print("\n" + "="*70)
    print(f"Test Results on: {folder_path.name}")
    print("="*70)
    print(f"\nTotal Images Processed: {processed_images}")
    print(f"Overall Accuracy: {acc:.4f} ({int(acc * processed_images)}/{processed_images})\n")
    
    print(f"{'Class':<15}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Support':<10}")
    print("-" * 70)
    for i, cls in enumerate(config.CLASS_NAMES):
        print(
            f"{cls:<15}{precision[i]:<12.4f}{recall[i]:<12.4f}"
            f"{f1[i]:<12.4f}{int(support[i]):<10}"
        )
    
    print("-" * 70)
    print(f"{'Macro Avg':<15}{precision_macro:<12.4f}{recall_macro:<12.4f}{f1_macro:<12.4f}")
    print("="*70 + "\n")
    
    # Full classification report
    report_dict = classification_report(
        y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))
    
    # Save results
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Save metrics as text
    report_path = output_dir / f"test_report_{folder_path.name}.txt"
    with open(report_path, "w") as f:
        f.write(f"Test Results on: {folder_path}\n")
        f.write("="*70 + "\n")
        f.write(f"Total Images: {processed_images}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(f"{'Class':<15}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Support':<10}\n")
        f.write("-" * 70 + "\n")
        for i, cls in enumerate(config.CLASS_NAMES):
            f.write(
                f"{cls:<15}{precision[i]:<12.4f}{recall[i]:<12.4f}"
                f"{f1[i]:<12.4f}{int(support[i]):<10}\n"
            )
        f.write("-" * 70 + "\n")
        f.write(f"{'Macro Avg':<15}{precision_macro:<12.4f}{recall_macro:<12.4f}{f1_macro:<12.4f}\n")
        f.write("="*70 + "\n\n")
        f.write(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, zero_division=0))
    
    print(f"✓ Report saved to: {report_path}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, config.CLASS_NAMES, output_dir / f"confusion_matrix_{folder_path.name}.png")
    
    # Plot per-class metrics
    plot_class_metrics(precision, recall, f1, config.CLASS_NAMES, output_dir / f"class_metrics_{folder_path.name}.png")
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "total_images": processed_images,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues", interpolation='nearest')
    
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10
            )
    
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Confusion matrix saved to: {save_path}")


def plot_class_metrics(precision, recall, f1, class_names, save_path):
    """Plot per-class metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(class_names))
    width = 0.6
    
    # Precision
    axes[0].bar(x, precision, width, color='skyblue')
    axes[0].set_ylabel('Precision', fontsize=11)
    axes[0].set_title('Precision by Class', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_ylim([0, 1.1])
    for i, v in enumerate(precision):
        axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    
    # Recall
    axes[1].bar(x, recall, width, color='lightgreen')
    axes[1].set_ylabel('Recall', fontsize=11)
    axes[1].set_title('Recall by Class', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_ylim([0, 1.1])
    for i, v in enumerate(recall):
        axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    
    # F1-Score
    axes[2].bar(x, f1, width, color='salmon')
    axes[2].set_ylabel('F1-Score', fontsize=11)
    axes[2].set_title('F1-Score by Class', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].set_ylim([0, 1.1])
    for i, v in enumerate(f1):
        axes[2].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Class metrics plot saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test model on a folder of images organized by class"
    )
    parser.add_argument(
        "--folder", 
        type=str, 
        required=True,
        help="Path to folder with class subfolders (e.g., datasets/split/test/)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="custom_cnn",
        choices=["custom_cnn"],
        help="Which model to use (default: custom_cnn)"
    )
    args = parser.parse_args()
    
    try:
        results = test_on_folder(args.folder, args.model)
        if results:
            print(f"\n✓ Test completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
