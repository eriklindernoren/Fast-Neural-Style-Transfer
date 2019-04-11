from models import TransformerNet
from utils import *
import torch
from torch.autograd import Variable
import argparse
import os
import tqdm
from PIL import Image
import skvideo.io

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to video")
    parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
    args = parser.parse_args()
    print(args)

    os.makedirs("images/outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = style_transform()

    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(args.checkpoint_model))
    transformer.eval()

    stylized_frames = []
    for frame in tqdm.tqdm(extract_frames(args.video_path), desc="Processing frames"):
        # Prepare input frame
        image_tensor = Variable(transform(frame)).to(device).unsqueeze(0)
        # Stylize image
        with torch.no_grad():
            stylized_image = transformer(image_tensor)
        # Add to frames
        stylized_frames += [deprocess(stylized_image)]

    # Create video from frames
    video_name = args.video_path.split("/")[-1].split(".")[0]
    writer = skvideo.io.FFmpegWriter(f"images/outputs/stylized-{video_name}.gif")
    for frame in tqdm.tqdm(stylized_frames, desc="Writing to video"):
        writer.writeFrame(frame)
    writer.close()
