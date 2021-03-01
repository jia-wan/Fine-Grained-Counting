from PIL import Image
from collections import OrderedDict
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2
import h5py
import json
import pdb


class FineGrainedDataset():
    def __init__(self, opt, stage):
        self.opt = opt
        if 'train' in stage:
            with open(opt.train_json, 'r') as outfile:
                self.data_list = json.load(outfile)
        elif 'val' in stage:
            with open(opt.val_json, 'r') as outfile:
                self.data_list = json.load(outfile)
        elif 'test' in stage:
            with open(opt.test_json, 'r') as outfile:
                self.data_list = json.load(outfile)

    def __getitem__(self, index):

        data = OrderedDict()
        img_path = self.data_list[index]
        #print(img_path)
        # read image
        img = Image.open(img_path).convert('RGB')
        if self.opt.gray:
            img = img.convert('L')
        ratio = min(1080/img.size[0], 1080/img.size[1])
        l = 32
        w, h = int(ratio*img.size[0]/l)*l, int(ratio*img.size[1]/l)*l
        o_w, o_h = img.size
        img = img.resize([w, h])

        #img = img.resize([int(ratio*img.size[0]), int(ratio*img.size[1])])

        ## read density map
        # get ground-truth path, dot, fix4, fix16, or adapt
        if 'fix4' in self.opt.dmap_type:
            temp = '_fix4.h5'
        elif 'fix16' in self.opt.dmap_type:
            temp = '_fix16.h5'
        elif 'adapt' in self.opt.dmap_type:
            temp = '_adapt.h5'
        elif 'dot' in self.opt.dmap_type:
            temp = '_dot.h5'
        else:
            print('dmap type error!')
        suffix = img_path[-4:]
        # suppose the ground-truth density maps are stored in ground-truth folder
        gt_path = img_path.replace(suffix, temp).replace('images', 'ground-truths')
        gt_file = h5py.File(gt_path, 'r')
        den = np.asarray(gt_file['density'])
        # reshape the dot map
        if 'dot' in self.opt.dmap_type:
            idx = den.nonzero()
            for i in range(len(idx[1])):
                idx[1][i] = int(idx[1][i] * h / o_h)
            for i in range(len(idx[2])):
                idx[2][i] = int(idx[2][i] * w / o_w)
            den = torch.zeros(2,h,w)
            den = np.array(den)
            den[idx] = 1

        # read roi mask
        if self.opt.roi:
            # get mask path
            # For Towards_vs_Away change the mask path
            mask_path = 'mask.h5'
            gt_file = h5py.File(mask_path, 'r')
            mask = np.asarray(gt_file['mask'])
            mask = torch.from_numpy(mask).unsqueeze(0)
            mask[mask > 0] = 1
            data['mask'] = mask

        # read semantic map
        if self.opt.smap:
            # get segmentation map
            seg_path = img_path.replace('.png', '_seg.h5')
            gt_file = h5py.File(seg_path, 'r')
            smap = np.asarray(gt_file['seg'])
            smap = torch.from_numpy(smap)#.unsqueeze(0)
            data['smap'] = smap


        # transformation
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])(img)
        den = torch.from_numpy(den)
        
        # return
        data['img'] = img
        data['den'] = den
        return data

    def __len__(self):
        return len(self.data_list)
