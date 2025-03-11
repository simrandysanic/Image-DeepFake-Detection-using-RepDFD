import os
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
import time
import shutil  

# Define dataset root
DATASET_PATH = r"D:\MTech - IIT Ropar\Subject Lessons\Semester 2\AI\Project\Image DeepFake Detection\WildDeep_Small"  # Use a shorter path

# Load CLIP preprocessing function
_, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")

# Define sets (train, test, valid)
sets = ["train", "test", "valid"]

# Function to check free disk space
def check_disk_space(min_space_gb=5):
    """Check if there is at least min_space_gb GB of free space (Windows-compatible)."""
    total, used, free = shutil.disk_usage(DATASET_PATH)
    free_space_gb = free / (1024 ** 3)  # Convert bytes to GB
    return free_space_gb >= min_space_gb

# Function to process images
def process_images(set_name, label):
    input_folder = os.path.join(DATASET_PATH, set_name, label)
    output_folder = os.path.join(DATASET_PATH, f"processed_{set_name}", label)
    os.makedirs(output_folder, exist_ok=True)

    for img_name in tqdm(os.listdir(input_folder), desc=f"Processing {set_name}/{label}"):
        img_path = os.path.join(input_folder, img_name)

        # Ensure it's an image file
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue  # Skip non-image files

        base_name, _ = os.path.splitext(img_name)  # Remove extension safely
        save_path = os.path.join(output_folder, base_name + ".pt")  # Save as .pt

        # Check if file already exists (to avoid reprocessing)
        if os.path.exists(save_path):
            continue

        # Try multiple times in case of temporary issues
        for attempt in range(3):  # Retry up to 3 times
            try:
                if not check_disk_space():
                    print("❌ Low disk space! Stopping processing.")
                    return
                
                img = Image.open(img_path).convert("RGB")  # Load image safely
                img = preprocess(img)  # Apply CLIP preprocessing
                torch.save(img, save_path)  # Save as PyTorch tensor
                break  # Success, exit retry loop

            except (IOError, OSError) as e:
                print(f"⚠️ Error processing {img_name} (Attempt {attempt + 1}/3): {e}")
                time.sleep(1)  # Wait a bit before retrying

        else:
            print(f"❌ Skipping {img_name} after 3 failed attempts.")

# Process images for all sets
for set_name in sets:
    process_images(set_name, "real")
    process_images(set_name, "fake")

print("✅ Preprocessing complete! Images saved as PyTorch tensors.")
