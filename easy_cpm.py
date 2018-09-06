import os
import re
import sys
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from network.rtpose_vgg import get_model
from network.post import decode_pose
from training.datasets.coco_data.preprocessing import (inception_preprocess,
                                              rtpose_preprocess,
                                              ssd_preprocess, vgg_preprocess)
from evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat


class CPM(object):
    """"""
    def __init__(self, model_fn, is_gpu=False):
        """"""
        # initiating the model
        self.model_fn = model_fn 
        self.model = get_model('vgg19')
        self.model.load_state_dict(torch.load(self.model_fn))
        self.model = torch.nn.DataParallel(self.model)

        self.is_gpu = is_gpu
        if self.is_gpu:
            self.model = self.model.cuda()
        
        self.model.float()
        self.model.eval()

    def run(self, img):
        """"""
        # Get results of original image
        multiplier = get_multiplier(img)

        with torch.no_grad():
            orig_paf, orig_heat = get_outputs(
                multiplier, img, self.model,  'rtpose')
                  
            # Get results of flipped image
            swapped_img = img[:, ::-1, :]
            flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img,
                                                    self.model, 'rtpose')

            # compute averaged heatmap and paf
            paf, heatmap = handle_paf_and_heat(
                orig_heat, flipped_heat, orig_paf, flipped_paf)
                    
        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        canvas, to_plot, candidate, subset = decode_pose(
            img, param, heatmap, paf)

        return candidate, subset, paf


if __name__ == "__main__":
    model = CPM('./network/weight/pose_model.pth')
    img = cv2.imread('./readme/ski.jpg')
    results = model.run(img)
