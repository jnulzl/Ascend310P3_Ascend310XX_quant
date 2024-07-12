#!/usr/bin/ python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

data process script.

"""
import os
import json
import sys
import numpy as np
import cv2 

BIN_ROOT_PATH = "./data/calibration"
VALID_IMG_NUM = 3072

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # return im, ratio, (dw, dh)
    return im


def prepare_image_input_gray(images, fixed_scale, des_height, des_width, mean, std):
    """Read image files to blobs [batch_size, 1, size, size]"""
    input_array = np.zeros((len(images), 1, des_height, des_width), np.float32)
    for index, im_file in enumerate(images):
        print(im_file)
        im_data = cv2.imread(im_file)
        im_data = im_data.astype(np.float32)
            
        if fixed_scale:
            im_data = letterbox(im_data, (des_height, des_width))
        else:
            im_data = cv2.resize(im_data, (des_width, des_height))

        input_array[index, 0] = (im_data - mean) / std

    return input_array
    
    
def prepare_image_input_color(images, fixed_scale, des_channels, des_height, des_width, is_BGR2RGB, means, stds):
    """Read image files to blobs [batch_size, des_channels, size, size]"""
    input_array = np.zeros((len(images), des_channels, des_height, des_width), np.float32)
    for index, im_file in enumerate(images):
        print(im_file)
        im_data = cv2.imread(im_file)
        im_data = im_data.astype(np.float32)
            
        if fixed_scale:
            im_data = letterbox(im_data, (des_height, des_width))
        else:
            im_data = cv2.resize(im_data, (des_width, des_height))
        
        if is_BGR2RGB:
            im_data = cv2.cvtColor(im_data, cv2.COLOR_BGR2RGB)
            
        for ch in range(des_channels):
            im_data[:,:,ch] = (im_data[:,:,ch] - means[ch]) / stds[ch]
                   
        input_array[index] = im_data.transpose(2, 0, 1) # HWC ->CHW

    return input_array


def prepare_image_input(images, fixed_scale, des_channels, des_height, des_width, is_BGR2RGB, means, stds):
    """Read image files to blobs [batch_size, des_channels, size, size]"""
    if 3 == des_channels:
        return prepare_image_input_color(images, fixed_scale, 3, des_height, des_width, is_BGR2RGB, means, stds)

    return prepare_image_input_gray(images, fixed_scale, des_height, des_width, means[0], stds[0])

def main():
    """process image and save it to bin"""
    if 2 != len(sys.argv):
        print("Usage:\n\t python %s json_path"%(sys.argv[0]))
        return
        
    json_path = sys.argv[1]
    with open(json_path,"r") as fpR:
        params_dict_all = json.load(fpR)
    
    params_dict = params_dict_all["det_pre_params"]

    img_root = params_dict["img_root"]
    fixed_scale = params_dict["fixed_scale"]
    batch_size = params_dict["batch_size"]
    des_channels = params_dict["des_channels"]
    des_height = params_dict["des_height"]
    des_width = params_dict["des_width"]
    is_BGR2RGB = params_dict["is_BGR2RGB"]
    means = params_dict["means"]
    stds = params_dict["stds"]
    
    image_paths = []
    tmps = os.listdir(img_root)
    for line in tmps:
        image_paths.append(os.path.join(img_root, line))
    
    valid_img_num = min(VALID_IMG_NUM, len(image_paths))
    input_array = prepare_image_input(image_paths[:valid_img_num], fixed_scale, des_channels, des_height, des_width, is_BGR2RGB, means, stds)

    bin_save_path = os.path.join(BIN_ROOT_PATH, os.path.basename(os.path.dirname(image_paths[0])) + "_batch%d"%(batch_size))
    if not os.path.exists(bin_save_path):
        os.makedirs(bin_save_path)

    count_batch = len(input_array) // batch_size
    for index in range(count_batch):
        begin_index = batch_size * index
        end_index = batch_size * (index + 1)
        input_array[begin_index:end_index].tofile(os.path.join(bin_save_path, "batch%d.bin"%(index)))


if __name__ == '__main__':
    main()
