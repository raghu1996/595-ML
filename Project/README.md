Repo containing the code used for the paper:
# Reimplementation of Noise2Noise and extension to dynamic scenes

## Noise2Noise
Made changes to 
1) validate across different noise sigma and plot PSNR
2) Increase model capacity

## Validate.ipynb 
Testing the Noise2Noise model on BSD300 and kodak dataset on a range of noise sigma

## Training_BSD300.ipynb 
Training the DnCNN with BSD300 dataset

## Optical flow.ipynb 
Code to calculate the optical flow between frames in a video

## blind_denoising.ipynb 
Video denoising by fine-tuning a pretrained network

## convert_video_to_frames.ipynb 
Snippet to convert a video file into several frames
