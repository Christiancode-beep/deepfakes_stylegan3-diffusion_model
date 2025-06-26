import torch
import sys
import pickle
from PIL import Image  # Correct import for Image module

# Add StyleGAN3 to Python path
sys.path.append('./stylegan3')

# 1. Load model
def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)['G_ema']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
G = load_pkl('models/stylegan3-r-256.pkl').to(device)
G.eval()

# 2. Generate image
z = torch.randn([1, G.z_dim]).to(device)
c = None  # Class label (None for unconditional generation)
img = G(z, c, truncation_psi=0.7)  # Generate image

# 3. Process and save image
img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0]
Image.fromarray(img.cpu().numpy()).save('output.png')  # Now will work correctly
print("Success! Check output.png")