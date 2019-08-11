from models import TransformerNet
from utils import *
import torch
from torch.autograd import Variable
import argparse
import os
from PIL import Image
import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
from test_on_image import Stylizer, save
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', '-v', type=str, required=True, help='Path to video')
    parser.add_argument('--checkpoint_model', '-c', type=str, required=True, help='Path to checkpoint model')
    parser.add_argument('--export_type', '-e', type=int, required=True, choices=[0,1,2], help='What to export? 0 for just frames, 1 for just video, 2 for both. 2 is recommended in case something goes wrong with the video.')
    parser.add_argument('--max_size', '-m', type=int, default=0, help='Tile size')
    parser.add_argument('--overlap', '-o', type=int, default=32, help='Overlap between tiles')
    parser.add_argument('--octave_num', '-on', type=int, default=None, help="Number of octaves")
    parser.add_argument('--octave_scale', '-os', type=float, default=1.4, help="Scale between octaves")
    args = parser.parse_args()
    print(args)



    stylizer = Stylizer(args.checkpoint_model)
    video_clip = VideoFileClip(args.video_path, audio=False)
    now = datetime.now()
    video_name = args.video_path.split("/")[-1].split(".")[0]
    out_dir = f"{now.year}{now.month}{now.day}-{now.hour}{now.minute}{now.second}-{args.checkpoint_model.split('/')[-1].split('.')[0]}-styled-{video_name}"
    os.makedirs(f"images/outputs/{out_dir}", exist_ok=True)

    # Create video from frames
    video_writer = None
    if args.export_type != 0:
        video_writer = ffmpeg_writer.FFMPEG_VideoWriter(f'images/outputs/{out_dir}/{video_name}.mp4', video_clip.size, video_clip.fps, codec="libx264",
                                                        preset="medium", bitrate="2000k",
                                                        audiofile=None, threads=None,
                                                        ffmpeg_params=None)

    try:
        fnum=0
        stylized_frames = []
        for frame in tqdm.tqdm(video_clip.iter_frames(), desc="Processing frames"):
            outframe = stylizer.stylize_with_octaves(frame, args.max_size, args.overlap, args.octave_num, args.octave_scale) if args.octave_num else stylizer.stylize_image(frame, args.max_size, args.overlap)
            if args.export_type != 0:
                video_writer.write_frame(outframe)
            if args.export_type != 1:
                save(f"{out_dir}/frame_{fnum}.jpg", np.asarray(outframe, dtype='float32'))
                save(f"{out_dir}/latest.jpg", np.asarray(outframe, dtype='float32')) #open image viewer on this to see video progress along
            fnum+=1

        video_writer.close()
    except KeyboardInterrupt:
        video_writer.close()
