from argparse import Namespace
import torch
import numpy as np
import cv2

from ImageMatchingToolbox.third_party.loftr.src.loftr import LoFTR as LoFTR_, default_cfg
from .base import Matching
from ImageMatchingToolbox.immatch.utils.data_io import img2tensor

class LoFTR(Matching):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)

        self.imsize = args.imsize        
        self.match_threshold = args.match_threshold
        self.no_match_upscale = args.no_match_upscale

        # Load model
        conf = dict(default_cfg)
        conf['match_coarse']['thr'] = self.match_threshold
        self.model = LoFTR_(config=conf)
        ckpt_dict = torch.load(args.ckpt)
        self.model.load_state_dict(ckpt_dict['state_dict'])
        self.model = self.model.eval().to(self.device)

        # Name the method
        self.ckpt_name = args.ckpt.split('/')[-1].split('.')[0]
        self.name = f'LoFTR_{self.ckpt_name}'        
        if self.no_match_upscale:
            self.name += '_noms'
        print(f'Initialize {self.name}')
        
    def load_im_gray(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img2tensor(gray, self.device, dfactor=8)

    def match_inputs_(self, gray1, gray2):
        batch = {'image0': gray1, 'image1': gray2}
        self.model(batch)
        kpts1 = batch['mkpts0_f'].cpu().numpy()
        kpts2 = batch['mkpts1_f'].cpu().numpy()
        scores = batch['mconf'].cpu().numpy()
        matches = np.concatenate([kpts1, kpts2], axis=1)
        return matches, kpts1, kpts2, scores

    def match_pairs(self, im1, im2):
        gray1, sc1 = self.load_im_gray(im1)
        gray2, sc1 = self.load_im_gray(im2)

        matches, kpts1, kpts2, scores = self.match_inputs_(gray1, gray2)

        return matches, kpts1, kpts2, scores
