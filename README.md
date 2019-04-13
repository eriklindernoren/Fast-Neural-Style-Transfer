# Fast Neural Style Transfer in PyTorch

<p align="center">
    <img src="assets/zurich.jpg" width="900"\>
</p>

PyTorch implementation of [Fast Neural Style Transfer](https://cs.stanford.edu/people/jcjohns/eccv16/) ([official Lua implementation](https://github.com/jcjohnson/fast-neural-style)).

I have made some trained models available [here](https://drive.google.com/drive/folders/1aRD6zakhcDImN2Y54qAT6f4801iLcCLB?usp=sharing).

## Train

```
python3 train.py  --dataset_path <path-to-dataset> \
                  --style_image <path-to-style-image> \
                  --epochs 1 \
                  --batch_size 4 \
                  --image_size 256
```

<p align="center">
    <img src="assets/celeba_mosaic.gif" width="400"\>
</p>
<p align="center">
    Figure: Training progress over the first 10,000 batches.
</p>

## Test on Video

```
python3 test_on_video.py  --video_path <path-to-video> \
                          --checkpoint_model <path-to-checkpoint> \
```

<p align="center">
    <img src="assets/stylized-celtics.gif" width="400"\>
</p>

## Test on Image

```
python3 test_on_image.py  --image_path <path-to-image> \
                          --checkpoint_model <path-to-checkpoint> \
```
