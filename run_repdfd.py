import os
import torch
import open_clip
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define dataset path
DATASET_PATH = r"D:\MTech - IIT Ropar\Subject Lessons\Semester 2\AI\Project\Image DeepFake Detection\WildDeep_Small"

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Function to load preprocessed images
def load_preprocessed_images(set_name, label):
    folder = os.path.join(DATASET_PATH, f"processed_{set_name}", label)
    images = []
    for file in tqdm(os.listdir(folder), desc=f"Loading {set_name}/{label} images"):
        if file.endswith(".pt"):
            img_tensor = torch.load(os.path.join(folder, file)).unsqueeze(0).to(device)
            images.append(img_tensor)
    return torch.cat(images, dim=0) if images else None  # Stack all images

# Load test set
real_images = load_preprocessed_images("test", "real")
fake_images = load_preprocessed_images("test", "fake")

# Prepare text prompts
text_inputs = tokenizer(["This is a real face", "This is a fake face"]).to(device)

# Function to calculate similarity scores
def get_clip_similarity(images):
    if images is None:
        return None
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_inputs)
        similarity = (image_features @ text_features.T).softmax(dim=-1)
    return similarity

# Get similarity scores
real_scores = get_clip_similarity(real_images)
fake_scores = get_clip_similarity(fake_images)

# Print average scores
if real_scores is not None and fake_scores is not None:
    print(f"\n✅ Real Image Avg Similarity: {real_scores.mean(dim=0)}")
    print(f"✅ Fake Image Avg Similarity: {fake_scores.mean(dim=0)}")

    # Convert tensors to numpy for plotting
    real_scores_np = real_scores[:, 0].cpu().numpy()  # Similarity to "real face"
    fake_scores_np = fake_scores[:, 1].cpu().numpy()  # Similarity to "fake face"

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(real_scores_np, bins=30, alpha=0.7, label="Real Face Similarity")
    plt.hist(fake_scores_np, bins=30, alpha=0.7, label="Fake Face Similarity")
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of CLIP Similarity Scores")
    plt.legend()
    plt.show()
