
from yolact import Yolact
from augmentations import BaseTransform, FastBaseTransform, Resize
from output_utils import postprocess, undo_image_transformation
from config import cfg, set_cfg, set_dataset, COLORS

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
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




def evalimage(net:Yolact):
    path = './frame.jpg'
    img = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(img.unsqueeze(0))
    dets_out = net(batch)


    img_gpu = img / 255.0
    h, w, _ = img.shape

    t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                        crop_masks        = True,
                                        score_threshold   = 0.05)    

    #score_threshold  0.15  得分阈值
    
    idx = t[1].argsort(0, descending=True)[:25]   #argsort 排序  提取top_k  选择前15个目标

    
    mask_cjf = np.zeros((h,w))
    Class_cjf = [t[0][idx].cpu().numpy()]   #每一层的类别

    for i in range(len(idx)):
        #print(Class_cjf[0][i])
        if Class_cjf[0][i] == 2:            #如果id等于某个类别
            #print(t[3][i].shape)    #torch.Size([415, 640])
            mask_cjf[t[3][i].cpu().numpy() == 1] =1
        if Class_cjf[0][i] == 5:            #如果id等于某个类别
            #print(t[3][i].shape)    #torch.Size([415, 640])
            mask_cjf[t[3][i].cpu().numpy() == 1] =1
        
    plt.imshow(mask_cjf)
    plt.title("Yolact")
    plt.show()


def showimage(net:Yolact):
    path = './frame.jpg'
    img = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(img.unsqueeze(0))
    dets_out = net(batch)

    img_gpu = img / 255.0
    h, w, _ = img.shape


    t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                        crop_masks        = True,
                                        score_threshold   = 0.05)  

    
    idx = t[1].argsort(0, descending=True)[:25]
        
    
    
    masks = t[3][idx]
    classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(25, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < 0.05:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        class_color=False
        color_idx = ( j * 5) % len(COLORS)

        color = COLORS[color_idx]
        return color




    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    if num_dets_to_consider == 0:
        return img_numpy

    for j in reversed(range(num_dets_to_consider)):
        x1, y1, x2, y2 = boxes[j, :]
        color = get_color(j)
        score = scores[j]

        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

        
        _class = cfg.dataset.class_names[classes[j]]
        text_str = '%s: %.2f' % (_class, score) 

        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

        text_pt = (x1, y1 - 3)
        text_color = [255, 255, 255]

        cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
        cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    plt.cla()
    plt.imshow(img_numpy)
    #plt.title("masks_draw")
    #plt.draw()
    plt.pause(0.03)
    #plt.close()



if __name__ == '__main__':



    with torch.no_grad():
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        dataset = None

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights('yolact_base_54_800000.pth')
        net.eval()  
        print(' Done.')

        net = net.cuda()
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False

        evalimage(net)


        while 1:
            if os.path.exists('frame.jpg'):
                #try:
                showimage(net)
                #except:
                #    print("****")
                time.sleep(0.03)
                

                



