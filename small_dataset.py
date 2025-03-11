import os
import shutil

# Define original and small dataset paths
DATASET_PATH = r"D:\MTech - IIT Ropar\Subject Lessons\Semester 2\AI\Project\Image DeepFake Detection\archive (1)"
SMALL_DATASET_PATH = r"D:\MTech - IIT Ropar\Subject Lessons\Semester 2\AI\Project\Image DeepFake Detection\WildDeep_Small"

# Define sets
sets = ["train", "test", "valid"]
labels = ["real", "fake"]
num_images = 500  # Limit per category

# Function to copy limited images
def copy_limited_images(set_name, label):
    src_folder = os.path.join(DATASET_PATH, set_name, label)
    dest_folder = os.path.join(SMALL_DATASET_PATH, set_name, label)
    os.makedirs(dest_folder, exist_ok=True)

    images = os.listdir(src_folder)[:num_images]  # Take first N images
    for img_name in images:
        shutil.copy(os.path.join(src_folder, img_name), os.path.join(dest_folder, img_name))

# Process all sets and labels
for set_name in sets:
    for label in labels:
        copy_limited_images(set_name, label)

print("✅ Small dataset created successfully!")
