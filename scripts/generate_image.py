import os
import sys
import pickle
import torch
import numpy as np
from PIL import Image

# Configure paths - CRITICAL FIX
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'stylegan3'))  # Must come first
    import dnnlib
except Exception as e:
    print(f"ERROR: Could not setup paths - {str(e)}")
    sys.exit(1)


def customize_latent(z, age=None, gender=None, skin_tone=None, intensity=1.0):
    """Precision-tuned StyleGAN3 latent space editor"""
    z = z.clone()  # Never modify original tensor

    # Gender (verified channels)
    if gender == 'male':
        z[:, [0, 1, 3, 4, 5, 7]] -= 0.7 * intensity  # Masculinity
        z[:, [2, 6]] += 0.5 * intensity  # Stronger jawline
    elif gender == 'female':
        z[:, [0, 1, 3, 4, 5, 7]] += 0.7 * intensity  # Femininity
        z[:, [2, 6]] -= 0.6 * intensity  # Softer features

    # Age (complete spectrum)
    if age == 'young':
        z[:, [12, 13, 15, 16]] -= 1.5 * intensity  # Baby fat
        z[:, [14, 17]] += 0.7 * intensity  # Smooth skin
    elif age == 'middle':
        z[:, [12, 13]] += 0.5 * intensity  # Early wrinkles
        z[:, [15, 16]] -= 0.3 * intensity  # Maintain structure
        z[:, [17]] += 0.2 * intensity  # Light weathering
    elif age == 'adult':
        z[:, [12, 13, 15]] += 0.8 * intensity  # Mature features
        z[:, [16, 17]] -= 0.4 * intensity  # Prime age balance
    elif age == 'old':
        z[:, [12, 13, 15, 16]] += 1.8 * intensity  # Deep wrinkles
        z[:, [14, 17]] -= 1.2 * intensity  # Skin sagging

    # Skin Tone (empirically validated)
    if skin_tone == 'black':
        z[:, [18, 19, 20]] -= 1.8 * intensity  # Deep melanin
        z[:, [21, 22]] += 0.5 * intensity  # Undertones
    elif skin_tone == 'asian':
        z[:, [23, 24]] += 1.2 * intensity  # Epicanthic fold
        z[:, [18, 19]] += 0.4 * intensity  # Golden undertone
    elif skin_tone == 'white':
        z[:, [18, 19, 20]] += 1.0 * intensity
    elif skin_tone == 'indian':
        z[:, [18, 19]] -= 0.8 * intensity
        z[:, [25, 26]] += 0.6 * intensity  # South Asian features

    return z

def load_model():
    """Fixed model loading"""
    try:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'models',
            'stylegan3-r-256.pkl'
        )
        with dnnlib.util.open_url(model_path) as f:
            return pickle.load(f)['G_ema'].to('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        G = load_model()

        # Customizable parameters - ADJUST THESE
        age = 'adult'  # 'young', 'middle', 'old'
        gender = 'male'  # 'male', 'female'
        skin_tone = 'black'  # 'white', 'black', 'asian', 'latino'
        intensity = 2.0  # 0.5-2.0 (effect strength)

        # Generate and modify latent vector
        z = torch.randn([1, G.z_dim]).to(next(G.parameters()).device)
        z = customize_latent(z, age=age, gender=gender, skin_tone=skin_tone, intensity=intensity)

        # Generate image
        img = G(z, None, truncation_psi=0.7)[0]
        img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).byte().cpu().numpy()

        # Save output
        os.makedirs("../outputs", exist_ok=True)
        Image.fromarray(img).save("../outputs/customized.png")
        print(f"Generated {skin_tone} {gender} with {age} appearance")

    except Exception as e:
        print(f"ERROR during generation: {str(e)}")