
import cv2
import numpy as np
from PIL import Image


import torch
import torchvision
import torch.nn as nn


class mtl_model(nn.Module):
    '''
    Multi-task model : Graph Scene Understanding and Captioning
    Forward uses features from feature_extractor
    '''
    def __init__(self, feature_extractor, scene_graph, caption):
        super(mtl_model, self).__init__()
        self.feature_extractor = feature_extractor
        self.scene_graph = scene_graph
        self.caption = caption
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    def forward(self, img_dir, det_boxes_all, caps_gt, node_num, spatial_feat, word2vec, roi_labels, val = False, text_field = None):               
        
        gsu_node_feat = None
        cp_node_feat = None

        # feature extraction model
        for index, img_loc in  enumerate(img_dir):
            _img = Image.open(img_loc).convert('RGB')
            _img = np.array(_img)
            
            img_stack = None
            for bndbox in det_boxes_all[index]:        
                roi = np.array(bndbox).astype(int)
                roi_image = _img[roi[1]:roi[3] + 1, roi[0]:roi[2] + 1, :]
                roi_image = self.transform(cv2.resize(roi_image, (224, 224), interpolation=cv2.INTER_LINEAR))
                roi_image = torch.autograd.Variable(roi_image.unsqueeze(0))
                img_stack = roi_image if img_stack == None else torch.cat((img_stack, roi_image))
            
            img_stack = img_stack.cuda()
            img_stack = self.feature_extractor(img_stack)
            
            # prepare graph node features  
            gsu_node_feat = img_stack.view(img_stack.size(0), -1) if gsu_node_feat == None else torch.cat((gsu_node_feat,img_stack.view(img_stack.size(0), -1)))
            
            # prepare caption node features
            if cp_node_feat == None:
                cp_node_feat = torch.unsqueeze(torch.cat((img_stack.view(img_stack.size(0), -1),torch.zeros((6-len(det_boxes_all[index])),512).cuda())),0)
            else: 
                cp_node_feat = torch.cat((cp_node_feat,torch.unsqueeze(torch.cat((img_stack.view(img_stack.size(0), -1),torch.zeros((6-len(det_boxes_all[index])),512).cuda())),0)))

        # Scene graph
        interaction = self.scene_graph(node_num, gsu_node_feat, spatial_feat, word2vec, roi_labels, validation= val)
        
        # caption model
        if val: 
            caption, _ = self.caption.beam_search(cp_node_feat, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
        else: 
            caption = self.caption(cp_node_feat, caps_gt)
        
        return interaction, caption