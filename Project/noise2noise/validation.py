# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import numpy as np
import PIL.Image
import tensorflow as tf

import dnnlib
import dnnlib.submission.submit as submit
import dnnlib.tflib.tfutil as tfutil
from dnnlib.tflib.autosummary import autosummary

import util
import config

class ValidationSet:
    def __init__(self, submit_config):
        self.images = None
        self.imagenames = None
        self.submit_config = submit_config
        return

    def load(self, dataset_dir):
        import glob
        abs_dirname = os.path.join(submit.get_path_from_template(dataset_dir), "*")
        #abs_dirname = os.path.join(submit.get_path_from_template(dataset_dir), "*/*")
        fnames = sorted(glob.glob(abs_dirname))
        if len(fnames) == 0:
            print ('\nERROR: No files found using the following glob pattern:', abs_dirname, '\n')
            sys.exit(1)

        images = []
        imagenames = []
        for fname in fnames:
            try:
                im = PIL.Image.open(fname).convert('RGB')
                arr = np.array(im, dtype=np.float32)
                reshaped = arr.transpose([2, 0, 1]) / 255.0 - 0.5
                images.append(reshaped)
                imagenames.append(fname)
            except OSError as e:
                print ('Skipping file', fname, 'due to error: ', e)
        self.images = images
        self.imagenames = imagenames
        
    def evaluate_video(self, net, iteration, noise_func):
        print("Inside evaluate_video")
        global avg_psnr 
        avg_psnr = 0.0
        for idx in range(len(self.images)):
            orig_img = self.images[idx]
            if "gt_" not in self.imagenames[idx]:
                continue
            w = orig_img.shape[2]
            h = orig_img.shape[1]

            #noisy_img = noise_func(orig_img)
            #noisy_img = orig_img.replace("gt_","",1)
            noisy_idx = self.imagenames.index(self.imagenames[idx].replace("gt_","",1))
            #print("GT image name is ",self.imagenames[idx])
            #print("Idx of noisy img is ",noisy_idx)
            #print("Noisy image name is ",self.imagenames[noisy_idx])
            noisy_img = self.images[noisy_idx]
            print("Noisy image shape: ",noisy_img.shape[2],noisy_img.shape[1])
            pred255 = util.infer_image(net, noisy_img)
            orig255 = util.clip_to_uint8(orig_img)
            print("Predicted image shape: ",pred255.shape[2],pred255.shape[1])
            print("orig image shape: ",w,h)
            assert (pred255.shape[2] == w and pred255.shape[1] == h)

            sqerr = np.square(orig255.astype(np.float32) - pred255.astype(np.float32))
            s = np.sum(sqerr)
            cur_psnr = 10.0 * np.log10((255*255)/(s / (w*h*3)))
            avg_psnr += cur_psnr

            util.save_image(self.submit_config, pred255, "img_{0}_val_{1}_pred.png".format(iteration, idx))

            if iteration == 0:
                util.save_image(self.submit_config, orig_img, "img_{0}_val_{1}_orig.png".format(iteration, idx))
                util.save_image(self.submit_config, noisy_img, "img_{0}_val_{1}_noisy.png".format(iteration, idx))
        avg_psnr /= len(self.images)
        print ('Average PSNR: %.2f' % autosummary('PSNR_avg_psnr', avg_psnr))
        return avg_psnr
        
    def evaluate_video_2(self, net, iteration, noise_func):
        print("Inside evaluate_video_2")
        global avg_psnr 
        avg_psnr = 0.0
        for idx in range(len(self.images)):
            orig_img = self.images[idx]
            if "original" not in self.imagenames[idx]:
                continue
            w = orig_img.shape[2]
            h = orig_img.shape[1]

            #noisy_img = noise_func(orig_img)
            #noisy_img = orig_img.replace("gt_","",1)
            noisy_idx = self.imagenames.index(self.imagenames[idx].replace("original","noisy",1))
            #print("GT image name is ",self.imagenames[idx])
            #print("Idx of noisy img is ",noisy_idx)
            #print("Noisy image name is ",self.imagenames[noisy_idx])
            noisy_img = self.images[noisy_idx]
            #print("Noisy image shape: ",noisy_img.shape[2],noisy_img.shape[1])
            pred255 = util.infer_image(net, noisy_img)
            orig255 = util.clip_to_uint8(orig_img)
            #print("Predicted image shape: ",pred255.shape[2],pred255.shape[1])
            #print("orig image shape: ",w,h)
            assert (pred255.shape[2] == w and pred255.shape[1] == h)

            sqerr = np.square(orig255.astype(np.float32) - pred255.astype(np.float32))
            s = np.sum(sqerr)
            cur_psnr = 10.0 * np.log10((255*255)/(s / (w*h*3)))
            #print("Curr PSNR is ",cur_psnr)
            avg_psnr += cur_psnr

            util.save_image(self.submit_config, pred255, "img_{0}_val_{1}_pred.png".format(iteration, idx))

            if iteration == 0:
                util.save_image(self.submit_config, orig_img, "img_{0}_val_{1}_orig.png".format(iteration, idx))
                util.save_image(self.submit_config, noisy_img, "img_{0}_val_{1}_noisy.png".format(iteration, idx))
        avg_psnr /= (len(self.images)/2)
        print ('Average PSNR: %.2f' % autosummary('PSNR_avg_psnr', avg_psnr))
        return avg_psnr
        
    def evaluate(self, net, iteration, noise_func):
        global avg_psnr 
        avg_psnr = 0.0
        for idx in range(len(self.images)):
            orig_img = self.images[idx]
            w = orig_img.shape[2]
            h = orig_img.shape[1]

            noisy_img = noise_func(orig_img)
            pred255 = util.infer_image(net, noisy_img)
            orig255 = util.clip_to_uint8(orig_img)
            assert (pred255.shape[2] == w and pred255.shape[1] == h)

            sqerr = np.square(orig255.astype(np.float32) - pred255.astype(np.float32))
            s = np.sum(sqerr)
            cur_psnr = 10.0 * np.log10((255*255)/(s / (w*h*3)))
            avg_psnr += cur_psnr

            util.save_image(self.submit_config, pred255, "img_{0}_val_{1}_pred.png".format(iteration, idx))

            if iteration == 0:
                util.save_image(self.submit_config, orig_img, "img_{0}_val_{1}_orig.png".format(iteration, idx))
                util.save_image(self.submit_config, noisy_img, "img_{0}_val_{1}_noisy.png".format(iteration, idx))
        avg_psnr /= len(self.images)
        print ('Average PSNR: %.2f' % autosummary('PSNR_avg_psnr', avg_psnr))
        return avg_psnr

def validate(submit_config: dnnlib.SubmitConfig, noise: dict, dataset: dict, network_snapshot: str):
    noise_augmenter = dnnlib.util.call_func_by_name(**noise)
    validation_set = ValidationSet(submit_config)
    validation_set.load(**dataset)

    ctx = dnnlib.RunContext(submit_config, config)

    tfutil.init_tf(config.tf_config)

    with tf.device("/gpu:0"):
        net = util.load_snapshot(network_snapshot)
        avg_psnr=validation_set.evaluate(net, 0, noise_augmenter.add_validation_noise_np)
        #avg_psnr=validation_set.evaluate_video(net, 0, noise_augmenter.add_validation_noise_np)
        #avg_psnr=validation_set.evaluate_video_2(net, 0, noise_augmenter.add_validation_noise_np)
        submit_config.psnr["avg_psnr"].append(avg_psnr)
        print("Average psnr printed in def validate is :")
        count = 0;
        for values in submit_config.psnr["avg_psnr"]:
            print(submit_config.psnr["stddev"][count],values)
            count = count+1
    ctx.close()

def infer_image(network_snapshot: str, image: str, out_image: str):
    tfutil.init_tf(config.tf_config)
    net = util.load_snapshot(network_snapshot)
    im = PIL.Image.open(image).convert('RGB')
    arr = np.array(im, dtype=np.float32)
    reshaped = arr.transpose([2, 0, 1]) / 255.0 - 0.5
    pred255 = util.infer_image(net, reshaped)
    t = pred255.transpose([1, 2, 0])  # [RGB, H, W] -> [H, W, RGB]
    PIL.Image.fromarray(t, 'RGB').save(os.path.join(out_image))
    print ('Inferred image saved in', out_image)
