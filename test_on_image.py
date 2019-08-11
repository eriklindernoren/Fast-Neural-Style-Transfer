from models import TransformerNet
from utils import *
import torch
from torch.autograd import Variable
import argparse
import os
import tqdm
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from imageio import imwrite
from math import floor
from datetime import datetime


class Stylizer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = style_transform()

        # Define model and load model checkpoint
        self.transformer = TransformerNet().to(self.device)
        self.transformer.load_state_dict(torch.load(model_path))
        self.transformer.eval()

    def get_index(self, x, tile, max_size, overlap):
        # Given a coordinate in the final image, what is the coordinate of the pixel
        # in the tile at that location
        return x if tile==0 else x-(tile*max_size)+overlap

    def do_section(self, section):
        # Prepare input
        image_tensor = Variable(self.transform(Image.fromarray(np.uint8(section)))).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)

        # Stylize image
        with torch.no_grad():
            stylized_array = deprocess(self.transformer(image_tensor))[:section.shape[0], :section.shape[1], :]

        return stylized_array

    def stylize_image(self, input_image, max_size=0, overlap=32):
        input_array = np.asarray(input_image, dtype='float32')
        if max_size != 0:
            width, height = input_array.shape[1::-1]
            wtiles = width // max_size + 1
            htiles = height // max_size + 1

            print('width: %i height: %i horizontal tiles: %i vertical tiles %i' % (width, height, wtiles, htiles))

            zeros = np.zeros(shape=(1,1), dtype='float32')

            arrayOfImages = [[zeros for x in range(wtiles)] for y in range(htiles)]

            x_domains = [[0, 0] for x in range(wtiles)]
            y_domains = [[0, 0] for y in range(htiles)]


            for j in range(htiles):
                for i in range(wtiles):
                    # boundaries of tile
                    x1 = max_size * i
                    x2 = min(width, max_size*(i+1))
                    y1 = max_size * j
                    y2 = min(height, max_size*(j+1))

                    # how many extra pixels on each side to compute
                    o_left = overlap if i > 0 else 0
                    o_right = overlap if i < wtiles-1 else 0
                    o_top = overlap if j > 0 else 0
                    o_bottom = overlap if j < htiles-1 else 0

                    # final boundaries of the window that will be computed
                    ox1 = x1 - o_left
                    ox2 = x2 + o_right
                    oy1 = y1 - o_top
                    oy2 = y2 + o_bottom

                    print(f'Tile({x1},{x2},{y1},{y2})')

                    # style transfer the current window
                    arrayOfImages[j][i] = self.do_section(input_array[oy1:oy2, ox1:ox2, :])

                    # store some boundaries for later. These are the coordinates in the
                    # final image within which only one tile will be used and without
                    # which will be a blend between multiple tiles.
                    if j==0:
                        x_domains[i][0] = x1 + o_left
                        x_domains[i][1] = x2 - o_right
                    if i==0:
                        y_domains[j][0] = y1 + o_top
                        y_domains[j][1] = y2 - o_bottom

            arrayOfRows = [zeros for y in range(htiles)]

            for j in range(htiles):
                row_array = np.zeros(shape=(arrayOfImages[j][0].shape[0], width, 3), dtype='float32')
                xfactors = np.zeros(shape=(width), dtype='float32')
                for x in range(width):
                    #find which tiles should be used
                    for i in range(len(x_domains)):
                        if i==len(x_domains)-1:
                            xfactors[x]=i
                        elif x >= x_domains[i][0] and x < x_domains[i+1][0]:
                            if x < x_domains[i][1]:
                                xfactors[x] = i
                            else:
                                xfactors[x] = (x-x_domains[i][1])/(2*overlap) + i
                            break

                    #if using only one tile just simply use it
                    if float(xfactors[x]).is_integer():
                        row_array[:, x, :] = arrayOfImages[j][int(xfactors[x])][:, self.get_index(x, int(floor(xfactors[x])), max_size, overlap), :]
                    #if using two tiles compute the factor for each tile and blend them
                    else:
                        muld_lower = arrayOfImages[j][int(floor(xfactors[x]))][:, self.get_index(x, int(floor(xfactors[x])), max_size, overlap), :] * (1-xfactors[x]%1)
                        muld_higher = arrayOfImages[j][int(floor(xfactors[x]+1))][:, self.get_index(x, int(floor(xfactors[x]+1)), max_size, overlap), :] * ((xfactors[x]%1))
                        row_array[:, x, :] = muld_lower + muld_higher
                arrayOfRows[j] = row_array


            final_image = np.zeros(shape=(height, width, 3), dtype='float32')
            yfactors = np.zeros(shape=(height), dtype='float32')
            for y in range(height):
                #find which tiles should be used
                for j in range(len(y_domains)):
                    if j==len(y_domains)-1:
                        yfactors[y]=j
                    elif y >= y_domains[j][0] and y < y_domains[j+1][0]:
                        if y < y_domains[j][1]:
                            yfactors[y] = j
                        else:
                            yfactors[y] = (y-y_domains[j][1])/(2*overlap) + j
                        break

                #if using only one row just simply use it
                if float(yfactors[y]).is_integer():
                    final_image[y, :, :] = arrayOfRows[int(floor(yfactors[y]))][self.get_index(y, int(floor(yfactors[y])), max_size, overlap), :, :]
                #if using two rows compute the factor for each row and blend them
                else:
                    muld_lower = arrayOfRows[int(floor(yfactors[y]))][self.get_index(y, int(floor(yfactors[y])), max_size, overlap), :, :] * (1-yfactors[y]%1)
                    muld_higher = arrayOfRows[int(floor(yfactors[y]+1))][self.get_index(y, int(floor(yfactors[y]+1)), max_size, overlap), :, :] * ((yfactors[y]%1))
                    final_image[y, :, :] = muld_lower + muld_higher

            return Image.fromarray(final_image.astype(np.uint8))
        else:
            return Image.fromarray(self.do_section(input_array).astype(np.uint8))

    def stylize_with_octaves(self, input_image, max_size=0, overlap=32, octaves=4, octave_scale=1.4):
        original = input_image
        input_image = scale_factor(input_image, (1/octave_scale) ** octaves)
        for octave in range(octaves):
            input_image = scale_factor(self.stylize_image(input_image, max_size, overlap), octave_scale)
            scaled_original = scale_absolute(original, input_image.size[0], input_image.size[1])
            input_image = Image.blend(input_image, scaled_original, 0.5)
        return self.stylize_image(input_image, max_size, overlap)


def save(name, array):
    Image.fromarray(array.astype(np.uint8)).save(f"images/outputs/{name}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-i', type=str, required=True, help="Path to image")
    parser.add_argument('--checkpoint_model', '-c', type=str, required=True, help="Path to checkpoint model")
    parser.add_argument('--max_size', '-m', type=int, default=0, help="Tile size")
    parser.add_argument('--overlap', '-o', type=int, default=32, help="Overlap between tiles")
    parser.add_argument('--octave_num', '-on', type=int, default=None, help="Number of octaves")
    parser.add_argument('--octave_scale', '-os', type=float, default=1.4, help="Scale between octaves")


    args = parser.parse_args()
    print(args)

    os.makedirs("images/outputs", exist_ok=True)

    stylizer = Stylizer(args.checkpoint_model)
    stylized_image = Image.open(args.image_path)
    if args.octave_num:
        stylized_image = stylizer.stylize_with_octaves(stylized_image, args.max_size, args.overlap, args.octave_num, args.octave_scale)
    else:
        stylized_image = stylizer.stylize_image(stylized_image, args.max_size, args.overlap)

    # Save image
    fn = args.image_path.split('/')[-1]
    now = datetime.now()
    print(fn)
    save(f"{now.year}{now.month}{now.day}-{now.hour}{now.minute}{now.second}-{args.checkpoint_model.split('/')[-1].split('.')[0]}-styled-{fn}", np.asarray(stylized_image, dtype='float32'))
