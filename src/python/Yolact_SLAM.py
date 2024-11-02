from yolact import Yolact
from augmentations import BaseTransform, FastBaseTransform, Resize
from output_utils import postprocess, undo_image_transformation
from config import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import argparse
import time
import random
import cProfile
import pickle
import json
import os

from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2




class Mask:

    """
    """

    def __init__(self):
        
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        dataset = None

        print('.....Load model.....')
        self.net = Yolact()
        self.net.load_weights('./src/python/yolact_base_54_800000.pth')
        self.net.eval()  

        self.net = self.net.cuda()
        self.net.detect.use_fast_nms = True
        self.net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False
        print('Done')




    def GetDynSeg(self,image,image2=None):
        path = "./src/python/frame.jpg"
        img = torch.from_numpy(cv2.imread(path)).cuda().float()
        batch = FastBaseTransform()(img.unsqueeze(0))
        dets_out = self.net(batch)


        h, w, _ = img.shape

        t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                        crop_masks        = True,
                                        score_threshold   = 0.15)    

        
        idx = t[1].argsort(0, descending=True)[:15]  
       

       
            
        mask_cjf = np.zeros((h,w))
        Class_cjf = [t[0][idx].cpu().numpy()]   

        for i in range(len(idx)):
            #print(Class_cjf[0][i])
            if Class_cjf[0][i] == 0:              #person
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1
            if Class_cjf[0][i] == 1:              #bicycle
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1
            if Class_cjf[0][i] == 2:              #car
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1
            if Class_cjf[0][i] == 3:              #motorcycle
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1
            if Class_cjf[0][i] == 4:              #airplane
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1
            if Class_cjf[0][i] == 5:              #bus
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1 
            if Class_cjf[0][i] == 6:              #train
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1
            if Class_cjf[0][i] == 7:              #truck
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1
            if Class_cjf[0][i] == 8:              #boat
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1
            if Class_cjf[0][i] == 14:             #bird
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1
            if Class_cjf[0][i] == 15:             #cat
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1
            if Class_cjf[0][i] == 16:             #dog
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1
            if Class_cjf[0][i] == 17:             #horse
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1
            if Class_cjf[0][i] == 18:             #sheep
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1
            if Class_cjf[0][i] == 19:             #cow
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1
            if Class_cjf[0][i] == 20:             #elephant
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1
            if Class_cjf[0][i] == 21:             #bear
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1
            if Class_cjf[0][i] == 22:             #zebra
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1 
            if Class_cjf[0][i] == 23:             #giraffe
                mask_cjf[t[3][i].cpu().detach().numpy() == 1] =1


        #plt.imshow(mask_cjf)
        #plt.title("mask_cjf")
        #plt.show()

        return mask_cjf














