# # !pip installe kaggle

# from dotenv import load_dotenv
# import os
# from pathlib import Path
# from kaggle.api.kaggle_api_extended import KaggleApi

# # Load .env
# load_dotenv()


# # Authenticate
# api = KaggleApi()
# api.authenticate()

# # Create folders
# DATA_DIR = Path("datasets")
# DATA_DIR.mkdir(exist_ok=True)

# # Download Garbage dataset
# api.dataset_download_files(
#     "sumn2u/garbage-classification-v2",
#     path=DATA_DIR / "garbage",
#     unzip=True
# )

# # Download TrashNet
# api.dataset_download_files(
#     "feyzazkefe/trashnet",
#     path=DATA_DIR / "trashnet",
#     unzip=True
# )

# print("Datasets downloaded successfully!")



# # from dotenv import load_dotenv
# # import os

# # load_dotenv()

# # print("KAGGLE_API_TOKEN:", os.getenv("KAGGLE_API_TOKEN"))
# # print("KAGGLE_USERNAME:", os.getenv("KAGGLE_USERNAME"))
# # print("KAGGLE_KEY:", os.getenv("KAGGLE_KEY"))




# from dotenv import load_dotenv
# import os
# import subprocess
# from pathlib import Path

# load_dotenv()

# token = os.getenv("KAGGLE_API_TOKEN")
# if not token:
#     raise ValueError("KAGGLE_API_TOKEN not found in .env")

# # Configure Kaggle's new authentication
# kaggle_dir = Path.home() / ".kaggle"
# kaggle_dir.mkdir(exist_ok=True)

# access_token_file = kaggle_dir / "access_token"
# access_token_file.write_text(token)
# os.chmod(access_token_file, 0o600)

# # Create dataset directories
# Path("datasets/garbage").mkdir(parents=True, exist_ok=True)
# Path("datasets/trashnet").mkdir(parents=True, exist_ok=True)

# # Download datasets using the CLI
# subprocess.run(
#     [
#         "kaggle",
#         "datasets",
#         "download",
#         "-d",
#         "sumn2u/garbage-classification-v2",
#         "-p",
#         "datasets/garbage",
#         "--unzip",
#     ],
#     check=True,
# )

# subprocess.run(
#     [
#         "kaggle",
#         "datasets",
#         "download",
#         "-d",
#         "feyzazkefe/trashnet",
#         "-p",
#         "datasets/trashnet",
#         "--unzip",
#     ],
#     check=True,
# )

# print("Datasets downloaded successfully!")




from dotenv import load_dotenv
import os
import subprocess
import shutil
from pathlib import Path

load_dotenv()

token = os.getenv("KAGGLE_API_TOKEN")
if not token:
    raise ValueError("KAGGLE_API_TOKEN not found in .env")

kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)
access_token_file = kaggle_dir / "access_token"
access_token_file.write_text(token)
os.chmod(access_token_file, 0o600)

raw_dir = Path("datasets_raw")
final_dir = Path("datasets/final")
(raw_dir / "garbage").mkdir(parents=True, exist_ok=True)
(raw_dir / "trashnet").mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = ["plastic", "paper", "metal", "glass", "organic"]
for c in TARGET_CLASSES:
    (final_dir / c).mkdir(parents=True, exist_ok=True)

# Download both datasets
subprocess.run(
    ["kaggle", "datasets", "download", "-d", "sumn2u/garbage-classification-v2",
     "-p", str(raw_dir / "garbage"), "--unzip"],
    check=True,
)
subprocess.run(
    ["kaggle", "datasets", "download", "-d", "feyzazkefe/trashnet",
     "-p", str(raw_dir / "trashnet"), "--unzip"],
    check=True,
)

def copy_images(src_folder: Path, dest_folder: Path, prefix: str):
    if not src_folder.exists():
        print(f"Skipping (not found): {src_folder}")
        return
    for i, img in enumerate(src_folder.glob("*")):
        if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
            shutil.copy(img, dest_folder / f"{prefix}_{i}{img.suffix.lower()}")

def find_class_dir(root: Path, class_name: str):
    matches = list(root.rglob(class_name))
    return matches[0] if matches else None

# --- garbage-classification-v2: has all 5 (biological -> organic) ---
mapping_v2 = {
    "plastic": "plastic",
    "paper": "paper",
    "metal": "metal",
    "glass": "glass",
    "biological": "organic",
}
for src_name, dest_name in mapping_v2.items():
    src = find_class_dir(raw_dir / "garbage", src_name)
    if src:
        copy_images(src, final_dir / dest_name, prefix="v2")
    else:
        print(f"Could not locate class folder for '{src_name}' in garbage-classification-v2")

# --- trashnet: supplement plastic/paper/metal/glass only (no organic) ---
for class_name in ["plastic", "paper", "metal", "glass"]:
    src = find_class_dir(raw_dir / "trashnet", class_name)
    if src:
        copy_images(src, final_dir / class_name, prefix="trashnet")
    else:
        print(f"Could not locate class folder for '{class_name}' in trashnet")

print("Done. Final dataset organized in:", final_dir.resolve())
for c in TARGET_CLASSES:
    n = len(list((final_dir / c).glob("*")))
    print(f"  {c}: {n} images")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     """
# Predict waste class for images using the trained custom CNN model.

# Usage:
#     # Single image
#     python predict.py --image path/to/image.jpg
    
#     # Folder of images (supports .jpg, .jpeg, .png)
#     python predict.py --folder path/to/images/
#     python predict.py --folder path/to/images/ --model custom_cnn
# """

# import argparse
# import torch
# from pathlib import Path
# from PIL import Image
# from torchvision import transforms
# import csv
# from datetime import datetime

# import config
# from train import build_model


# def predict_single_image(image_path: str, model_name: str = "custom_cnn"):
#     """
#     Load model checkpoint and predict class for a single image.
    
#     Args:
#         image_path: Path to the image file
#         model_name: Name of the model (default: custom_cnn)
    
#     Returns:
#         Tuple of (class_name, confidence_score)
#     """
#     device = config.DEVICE
    
#     # Load model
#     model = build_model(model_name).to(device)
#     ckpt_path = Path(config.CHECKPOINT_DIR) / f"{model_name}_best.pt"
    
#     if not ckpt_path.exists():
#         raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
#     model.load_state_dict(torch.load(ckpt_path, map_location=device))
#     model.eval()
#     print(f"✓ Loaded model from: {ckpt_path}")
    
#     # Load and transform image
#     if not Path(image_path).exists():
#         raise FileNotFoundError(f"Image not found at {image_path}")
    
#     image = Image.open(image_path).convert("RGB")
#     print(f"✓ Loaded image: {image_path} (size: {image.size})")
    
#     transform = transforms.Compose([
#         transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
#     ])
#     image_tensor = transform(image).unsqueeze(0).to(device)
    
#     # Predict
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         probs = outputs.softmax(dim=1)[0]
#         confidence, pred_idx = probs.max(dim=0)
    
#     pred_class = config.CLASS_NAMES[pred_idx.item()]
#     confidence_val = confidence.item()
    
#     # Print results
#     print(f"\n{'='*50}")
#     print(f"Predicted Class: {pred_class}")
#     print(f"Confidence: {confidence_val:.2%}")
#     print(f"{'='*50}\n")
    
#     # Show all probabilities
#     print("Class Probabilities:")
#     for i, class_name in enumerate(config.CLASS_NAMES):
#         prob = probs[i].item()
#         bar = "█" * int(prob * 30)
#         print(f"  {class_name:<10} {prob:.2%} {bar}")
    
#     return pred_class, confidence_val


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Predict waste class for an image")
#     parser.add_argument(
#         "--image", 
#         type=str, 
#         required=True,
#         help="Path to the image file"
#     )
#     parser.add_argument(
#         "--model", 
#         type=str, 
#         default="custom_cnn",
#         choices=["custom_cnn"],
#         help="Which model to use (default: custom_cnn)"
#     )
#     args = parser.parse_args()
    
#     try:
#         predict_single_image(args.image, args.model)
#     except Exception as e:
#         print(f"❌ Error: {e}")
