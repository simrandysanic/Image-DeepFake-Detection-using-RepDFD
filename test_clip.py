import torch
import open_clip

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)

# Print confirmation
print("CLIP Model Loaded Successfully on:", device)
