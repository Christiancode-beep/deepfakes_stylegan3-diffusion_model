import os
import sys
import pickle
import cv2
import torch
import time
from PIL import Image

# Configure paths
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.extend([PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'stylegan3')])
    import dnnlib
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'stylegan3-r-256.pkl')

except Exception as e:
    print(f"ERROR: Path setup failed - {str(e)}")
    sys.exit(1)


def customize_latent(z, age=None, gender=None, intensity=1.0):
    """
    Modify latent vector based on desired attributes
    intensity: 0.0 (neutral) to 2.0 (strong)
    """
    if gender == 'male':
        z[:, :8] -= 1.0 * intensity  # Masculinize
    elif gender == 'female':
        z[:, :8] += 1.0 * intensity  # Feminize

    if age == 'old':
        z[:, 12:20] += 1.5 * intensity  # Aging
    elif age == 'young':
        z[:, 12:20] -= 1.0 * intensity

    return z


def main():
    """Customizable video generation workflow"""
    try:
        # 1. Load model
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file missing at {MODEL_PATH}")
            return

        with open(MODEL_PATH, 'rb') as f:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            G = pickle.load(f)['G_ema'].to(device)
            print("Model loaded successfully")

        # 2. Video configuration - ADJUST THESE VALUES
        duration = 3  # seconds
        fps = 3  # Reduced for slower playback
        render_delay = 0.7  # Seconds between frames (0 = fastest)
        age = 'old'  # None, 'young', 'middle', 'old'
        gender = 'male'  # None, 'male', 'female'

        # 3. Generate video
        frames = []
        for i in range(duration * fps):
            # Create and customize latent vector
            z = torch.randn([1, G.z_dim]).to(device) * (i / (duration * fps))
            z = customize_latent(z, age=age, gender=gender)

            # Generate frame
            with torch.no_grad():
                img = G(z, None, truncation_psi=0.7)[0]
                img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).byte().cpu().numpy()
                frames.append(img[:, :, ::-1])  # RGB to BGR

            time.sleep(render_delay)  # Control rendering speed

        # 4. Save output
        output_dir = "../outputs"
        os.makedirs(output_dir, exist_ok=True)

        if len(frames) > 0:
            height, width = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                os.path.join(output_dir, 'video.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height))

            for frame in frames:
                writer.write(frame)
            writer.release()
            print(f"SUCCESS: {duration}s video saved to {output_dir}")
        else:
            print("ERROR: No frames generated")

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()