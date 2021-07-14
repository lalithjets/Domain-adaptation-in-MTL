import os
import cv2
import random

import json
import h5py
import itertools
import numpy as np
from PIL import Image
import argparse, pickle

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import torchvision.models
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader as TorchDataLoader

# caption libraries
import evaluation
import collections
from data.utils import nostdout
from data.example import Example
from data.field import TextField, RawField
from models.transformer import MemoryAugmentedEncoder_CBS
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory

# graph libraries
from utils.g_vis_img import *
from models.graph_su import *
from evaluation.graph_eval import *

# feature extractor
from models.feature_extractor import *

# Random seeds
seed = 27
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class SurgicalSceneConstants():
    '''
    Surgical Scene constants
    '''
    def __init__( self):
        self.instrument_classes = ( 'kidney', 'bipolar_forceps', 'prograsp_forceps', 'large_needle_driver',
                                'monopolar_curved_scissors', 'ultrasound_probe', 'suction', 'clip_applier',
                                'stapler', 'maryland_dissector', 'spatulated_monopolar_cautery')
        self.action_classes = ( 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation', 
                                'Tool_Manipulation', 'Cutting', 'Cauterization', 
                                'Suction', 'Looping', 'Suturing', 'Clipping', 'Staple', 
                                'Ultrasound_Sensing')


class CELossWithLS(torch.nn.Module):
    '''
    label smoothing cross-entropy loss for captioning
    '''
    def __init__(self, classes=None, smoothing=0.1, gamma=3.0, isCos=True, ignore_index=-1):
        super(CELossWithLS, self).__init__()
        self.complement = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        with torch.no_grad():
            oh_labels = F.one_hot(target.to(torch.int64), num_classes = self.cls).permute(0,1,2).contiguous()
            smoothen_ohlabel = oh_labels * self.complement + self.smoothing / self.cls

        logs = self.log_softmax(logits[target!=self.ignore_index])
        pt = torch.exp(logs)
        return -torch.sum((1-pt).pow(self.gamma)*logs * smoothen_ohlabel[target!=self.ignore_index], dim=1).mean()


class DataLoader(TorchDataLoader):
    '''
    Custom dataloader
    '''
    def __init__(self, dataset, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, *args, collate_fn=dataset.collate_fn(), **kwargs)

class Dataset(object):
    '''
    Custom Dataset to process dataset for both graph scene understanding and caption generation.
    '''
    def __init__(self, examples, fields, gsu_const):
        self.examples = examples
        self.fields = dict(fields)
        self.file_dir = gsu_const['file_dir']
        self.img_dir = gsu_const['img_dir']
        self.dataconst = gsu_const['dataconst']
        self.feature_extractor = gsu_const['feature_extractor']
        self.word2vec = h5py.File(gsu_const['w2v_loc'], 'r')
        
    # word2vec
    def _get_word2vec(self,node_ids):
        word2vec = np.empty((0,300))
        for node_id in node_ids:
            vec = self.word2vec[self.dataconst.instrument_classes[node_id]]
            word2vec = np.vstack((word2vec, vec))
        return word2vec

    def __getitem__(self, i):
        example = self.examples[i]
        frame_path = getattr(example, 'image')
        frame_path = frame_path.split("/")
        _img_loc = os.path.join(self.file_dir, frame_path[0],self.img_dir,frame_path[-1].split("_")[0]+'.png')
        frame_data = h5py.File(os.path.join(self.file_dir, frame_path[0],'vsgat',self.feature_extractor, frame_path[-1].split("_")[0]+'_features.hdf5'), 'r')    

        # caption data
        cp_data = []
        for field_name, field in self.fields.items():
            if field_name == 'image' and field == None: cp_data.append(np.zeros((6,512), dtype = np.float32))
            else: cp_data.append(field.preprocess(getattr(example, field_name)))   
        if len(cp_data) == 1: cp_data = cp_data[0]
        
        # graph data
        gsu_data = {}
        gsu_data['img_name'] = frame_data['img_name'].value[:] + '.jpg'
        gsu_data['img_loc'] = _img_loc
        gsu_data['node_num'] = frame_data['node_num'].value
        gsu_data['roi_labels'] = frame_data['classes'][:]
        gsu_data['det_boxes'] = frame_data['boxes'][:]
        gsu_data['edge_labels'] = frame_data['edge_labels'][:]
        gsu_data['edge_num'] = gsu_data['edge_labels'].shape[0]
        gsu_data['spatial_feat'] = frame_data['spatial_features'][:]
        gsu_data['word2vec'] = self._get_word2vec(gsu_data['roi_labels'])
        if self.fields['image'] == None: gsu_data['features'] = np.zeros((gsu_data['node_num'],512), dtype = np.float32)
        else: gsu_data['features'] = frame_data['node_features'][:]
        
        data = {}
        data['cp_data'] = cp_data
        data['gsu_data'] = gsu_data
        return data

    def __len__(self):
        return len(self.examples)

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)
                
    def collate_fn(self):
        def collate(batch):
            gsu_batch_data = {}
            gsu_batch_data['img_name'] = []
            gsu_batch_data['img_loc'] = []
            gsu_batch_data['node_num'] = []
            gsu_batch_data['roi_labels'] = []
            gsu_batch_data['det_boxes'] = []
            gsu_batch_data['edge_labels'] = []
            gsu_batch_data['edge_num'] = []
            gsu_batch_data['features'] = []
            gsu_batch_data['spatial_feat'] = []
            gsu_batch_data['word2vec'] = []

            for data in batch:
                gsu_batch_data['img_name'].append(data['gsu_data']['img_name'])
                gsu_batch_data['img_loc'].append(data['gsu_data']['img_loc'])
                gsu_batch_data['node_num'].append(data['gsu_data']['node_num'])
                gsu_batch_data['roi_labels'].append(data['gsu_data']['roi_labels'])
                gsu_batch_data['det_boxes'].append(data['gsu_data']['det_boxes'])
                gsu_batch_data['edge_labels'].append(data['gsu_data']['edge_labels'])
                gsu_batch_data['edge_num'].append(data['gsu_data']['edge_num'])
                gsu_batch_data['features'].append(data['gsu_data']['features'])
                gsu_batch_data['spatial_feat'].append(data['gsu_data']['spatial_feat'])
                gsu_batch_data['word2vec'].append(data['gsu_data']['word2vec'])

            gsu_batch_data['edge_labels'] = torch.FloatTensor(np.concatenate(gsu_batch_data['edge_labels'], axis=0))
            gsu_batch_data['features'] = torch.FloatTensor(np.concatenate(gsu_batch_data['features'], axis=0))
            gsu_batch_data['spatial_feat'] = torch.FloatTensor(np.concatenate(gsu_batch_data['spatial_feat'], axis=0))
            gsu_batch_data['word2vec'] = torch.FloatTensor(np.concatenate(gsu_batch_data['word2vec'], axis=0))
            
            cp_batch_data = []
            tensors = []
            
            for data in batch: cp_batch_data.append(data['cp_data'])
            if len(self.fields) == 1: cp_batch_data = [cp_batch_data, ]
            else: cp_batch_data = list(zip(*cp_batch_data))

            for field, data in zip(self.fields.values(), cp_batch_data):
                if field == None: tensor = default_collate(data)
                else: tensor = field.process(data)
                if isinstance(tensor, collections.Sequence) and any(isinstance(t, torch.Tensor) for t in tensor):
                    tensors.extend(tensor)
                else: tensors.append(tensor)

            if len(tensors) > 1:cp_batch_data = tensors
            else: cp_batch_data = tensors[0]
            
            batch_data = {}
            batch_data['gsu'] = gsu_batch_data
            batch_data['cp'] = cp_batch_data
            
            return(batch_data)

        return collate

class PairedDataset(Dataset):
    def __init__(self, examples, fields, gsu_const):
        assert ('image' in fields)
        assert ('text' in fields)
        super(PairedDataset, self).__init__(examples, fields, gsu_const)
        self.image_field = self.fields['image']
        if self.image_field == None: print('no pre-extracted image featured')
        self.text_field = self.fields['text']
        
    def image_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = Dataset(self.examples, fields, gsu_const)
        return dataset
        
class MTL_DATASET(PairedDataset):
    def __init__(self, image_field, text_field, gsu_const, img_root, ann_root, id_root=None):
        # setting training and val root
        roots = {}
        roots['train'] = { 'img': img_root, 'cap': os.path.join(ann_root, 'captions_train.json')}
        roots['val'] = {'img': img_root, 'cap': os.path.join(ann_root, 'captions_val.json')}

        # Getting the id: planning to remove this in future
        if id_root is not None:
            ids = {}
            ids['train'] = json.load(open(os.path.join(id_root, 'WithCaption_id_path_train.json'), 'r'))
            ids['val'] = json.load(open(os.path.join(id_root, 'WithCaption_id_path_val.json'), 'r'))   
        else: ids = None
        
        with nostdout():
            self.train_examples, self.val_examples = self.get_samples(roots, ids)
        examples = self.train_examples + self.val_examples
        super(MTL_DATASET, self).__init__(examples, {'image': image_field, 'text': text_field}, gsu_const)   

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields, gsu_const) 
        val_split = PairedDataset(self.val_examples, self.fields, gsu_const)
        return train_split, val_split

    @classmethod
    def get_samples(cls, roots, ids_dataset=None):
        train_samples = []
        val_samples = []
   
        for split in ['train', 'val']:
            anns = json.load(open(roots[split]['cap'], 'r'))
            if ids_dataset is not None: ids = ids_dataset[split]
                
            for index in range(len(ids)):              
                id_path = ids[index]
                caption = anns[index]['caption']
                example = Example.fromdict({'image': os.path.join('', id_path), 'text': caption})
                if split == 'train': train_samples.append(example)
                elif split == 'val': val_samples.append(example)
                    
        return train_samples, val_samples


class mtl_model(nn.Module):
    '''
    Multi-task model : Graph Scene Understanding and Captioning
    Forward uses features from feature_extractor
    '''
    def __init__(self, feature_extractor, graph, caption):
        super(mtl_model, self).__init__()
        self.feature_extractor = feature_extractor
        self.graph_su = graph
        self.caption = caption
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    def forward(self, img_dir, det_boxes_all, caps_gt, node_num, features, spatial_feat, word2vec, roi_labels, val = False, text_field = None):               
        
        gsu_node_feat = None
        cp_node_feat = None

        # feature extraction model
        for index, img_loc in  enumerate(img_dir):
            _img = Image.open(img_loc).convert('RGB')
            _img = np.array(_img)
            img_stack = None
            for idx, bndbox in enumerate(det_boxes_all[index]):        
                roi = np.array(bndbox).astype(int)
                roi_image = _img[roi[1]:roi[3] + 1, roi[0]:roi[2] + 1, :]
                roi_image = self.transform(cv2.resize(roi_image, (224, 224), interpolation=cv2.INTER_LINEAR))
                roi_image = torch.autograd.Variable(roi_image.unsqueeze(0))
                # stack nodes images per image
                if img_stack is None: img_stack = roi_image
                else: img_stack = torch.cat((img_stack, roi_image))
            
            img_stack = img_stack.cuda()
            # send the stack to feature extractor
            vis_feature = self.feature_extractor(img_stack)
            vis_feature = vis_feature.view(vis_feature.size(0), -1)
            
            if gsu_node_feat == None: gsu_node_feat = vis_feature
            else: gsu_node_feat = torch.cat((gsu_node_feat,vis_feature))
            
            vis_feature = torch.unsqueeze(torch.cat((vis_feature,torch.zeros((6-len(vis_feature)),512).cuda())),0)
            if cp_node_feat == None: cp_node_feat = vis_feature
            else: cp_node_feat = torch.cat((cp_node_feat,vis_feature))
    
        # caption model
        if val == True: caption_output, _ = self.caption.beam_search(cp_node_feat, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
        else: caption_output = self.caption(cp_node_feat, caps_gt)
        
        # graph su model
        interaction = self.graph_su(node_num, gsu_node_feat, spatial_feat, word2vec, roi_labels, validation= val)
        
        return interaction, caption_output


def build_model(args, text_field, device):
    '''
    Build MTL model
    1) Feature Extraction
    2) Caption Model
    3) Graph Scene Understanding Model
    '''

    ''' ==== caption model ===='''
    # caption encoder
    if args.cp_cbs == 'True':encoder = MemoryAugmentedEncoder_CBS(3, 0, attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': args.m})
    else: encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': args.m}) 
    # caption decoder
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    # caption model
    caption_model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.cp_cbs == 'True': caption_model.encoder.get_new_kernels(0, args.cp_kernel_sizex, args.cp_kernel_sizey, args.cp_decay_epoch, args.cp_std_factor, args.cp_cbs_filter)
    # caption load pre-trained weights
    pretrained_model = torch.load(args.cp_checkpoint+('%s_best.pth' % args.exp_name))
    caption_model.load_state_dict(pretrained_model['state_dict']) 

    '''==== graph model ===='''
    # graph model
    graph_su_model = AGRNN(bias= True, bn= False, dropout=0.3, multi_attn=False, layer=1, diff_edge=False, use_cbs = args.gsu_cbs)
    if args.gsu_cbs: graph_su_model.grnn1.gnn.apply_h_h_edge.get_new_kernels(0)
    # graph load pre-trained weights
    pretrained_model = torch.load(args.gsu_checkpoint)
    graph_su_model.load_state_dict(pretrained_model['state_dict'])
    #graph_su_model.eval()

    '''==== Feature extractor ===='''
    # feature extraction model
    if args.fe_use_SC: feature_network = SupConResNet(args=args)
    else: feature_network = ResNet18(args)
    if args.fe_use_cbs:
        if args.fe_use_SC: feature_network.encoder.get_new_kernels(0)
        else: feature_network.get_new_kernels(0)
    # based on cuda
    num_gpu = torch.cuda.device_count()
    if num_gpu > 0:
        device_ids = np.arange(num_gpu).tolist()    
        if args.fe_use_SC: feature_network.encoder = nn.DataParallel(feature_network.encoder) #feature_network = feature_network.cuda()
        else: feature_network = nn.DataParallel(feature_network, device_ids=device_ids)
    # feature extraction pre-trained weights
    feature_network.load_state_dict(torch.load(args.fe_modelpath))
    # extract the encoder layer
    if args.fe_use_SC: feature_network = feature_network.encoder
    else:
        if args.fe_use_cbs: feature_network = nn.Sequential(*list(feature_network.module.children())[:-2])
        else: feature_network = nn.Sequential(*list(feature_network.module.children())[:-1])

    model = mtl_model(feature_network, graph_su_model, caption_model)
    model = model.to(device)
    return model


def eval_mtl(model, dataloader, text_field):
    '''
    Evaluate MTL
    '''
    
    gen = {}
    gts = {}

    model.eval()

    # graph
    g_criterion = nn.MultiLabelSoftMarginLoss()                   
    g_edge_count = 0
    g_total_acc = 0.0
    g_total_loss = 0.0
    g_logits_list = []
    g_labels_list = []
    
    for it, data in tqdm(enumerate(iter(dataloader))):
            
        graph_data = data['gsu']
        cp_data = data['cp']
            
        # graph
        #img_name = graph_data['img_name']
        #edge_num = graph_data['edge_num']
        img_loc = graph_data['img_loc']
        node_num = graph_data['node_num']
        roi_labels = graph_data['roi_labels']
        det_boxes = graph_data['det_boxes']
        edge_labels = graph_data['edge_labels']
        features = graph_data['features']
        spatial_feat = graph_data['spatial_feat']
        word2vec = graph_data['word2vec']
        features, spatial_feat, word2vec, edge_labels = features.to(device), spatial_feat.to(device), word2vec.to(device), edge_labels.to(device)         
        
        _, caps_gt = cp_data
            
        with torch.no_grad():              
    
            g_output, caption_out = model(img_loc, det_boxes, caps_gt, node_num, features, spatial_feat, word2vec, roi_labels, val = True, text_field = text_field)
        
            g_logits_list.append(g_output)
            g_labels_list.append(edge_labels)
            # loss and accuracy
            g_loss = g_criterion(g_output, edge_labels.float())
            g_acc = np.sum(np.equal(np.argmax(g_output.cpu().data.numpy(), axis=-1), np.argmax(edge_labels.cpu().data.numpy(), axis=-1)))
            
        # accumulate loss and accuracy of the batch
        g_total_loss += g_loss.item() * edge_labels.shape[0]
        g_total_acc  += g_acc
        g_edge_count += edge_labels.shape[0]
        
        caps_gen = text_field.decode(caption_out, join_words=False)
        
        for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
            gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
            gen['%d_%d' % (it, i)] = [gen_i, ]    
            gts['%d_%d' % (it, i)] = [gts_i,]
        
    #graph evaluation
    g_total_acc = g_total_acc / g_edge_count
    g_total_loss = g_total_loss / len(dataloader)

    g_logits_all = torch.cat(g_logits_list).cuda()
    g_labels_all = torch.cat(g_labels_list).cuda()
    g_logits_all = F.softmax(g_logits_all, dim=1)
    g_map_value, g_ece, g_sce, g_tace, g_brier, g_uce = calibration_metrics(g_logits_all, g_labels_all, 'test')

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)

    scores, _ = evaluation.compute_scores(gts, gen)
    print('Graph : {acc: %0.6f map: %0.6f loss: %0.6f, ece:%0.6f, sce:%0.6f, tace:%0.6f, brier:%.6f, uce:%.6f}' %(g_total_acc, g_map_value, g_total_loss, g_ece, g_sce, g_tace, g_brier, g_uce.item()) )
    print(print("Caption Scores :", scores))


def train(epoch, lrc, model, dataloader, dict_dataloader_val, text_field):
    '''
    Finding optimal temperature scale for graph scene understanding task
    '''
    
    

    #if args.optim == 'sgd': 
    optimizer = optim.SGD(model.feature_extractor.parameters(), lr= lrc, momentum=0.9, weight_decay=0)
    #else: optimizer = optim.Adam(model.parameters(), lr= lrc, weight_decay=0)
       
    g_criterion = nn.MultiLabelSoftMarginLoss()
    
    for epoch_count in range(epoch):
        
        model.train()
        print("=========== Train ===============")
        running_loss = 0.0
        running_g_acc = 0.0
        running_edge_count = 0
        iters = 0
        
        for it, data in tqdm(enumerate(iter(dataloader))):
            iters += 1
            
            graph_data = data['gsu']
            cp_data = data['cp']
            
            # graph
            # img_name = graph_data['img_name']
            # edge_num = graph_data['edge_num']
            img_loc = graph_data['img_loc']
            node_num = graph_data['node_num']
            roi_labels = graph_data['roi_labels']
            det_boxes = graph_data['det_boxes']
            edge_labels = graph_data['edge_labels']
            features = graph_data['features']
            spatial_feat = graph_data['spatial_feat']
            word2vec = graph_data['word2vec']
            features, spatial_feat, word2vec, edge_labels = features.to(device), spatial_feat.to(device), word2vec.to(device), edge_labels.to(device)    
            
            # caption
            caption_nodes, caps_gt = cp_data
            caption_nodes, caps_gt = caption_nodes.to(device), caps_gt.to(device)
            
            model.zero_grad()
            
            interaction, caption_output = model( img_loc, det_boxes, caps_gt, node_num, features, spatial_feat, word2vec, roi_labels, val = False)
            
            # graph loss and acc
            interaction = F.softmax(interaction, dim=1)
            g_loss = g_criterion(interaction, edge_labels.float())
            g_acc = np.sum(np.equal(np.argmax(interaction.cpu().data.numpy(), axis=-1), np.argmax(edge_labels.cpu().data.numpy(), axis=-1)))
                    
            # caption loss
            c_criterion = CELossWithLS(classes=len(text_field.vocab), smoothing=0.1, gamma=0.0, isCos=False, ignore_index=text_field.vocab.stoi['<pad>'])
            c_loss = c_criterion(caption_output[:, :-1].contiguous(), caps_gt[:, 1:].contiguous())
            
            #uda:
            loss = (0.5 * g_loss) + (0.5 * c_loss)
            #uda_graph:
            #loss = g_loss
            #uda_caption:
            #loss = c_loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_g_acc += g_acc
            running_edge_count += edge_labels.shape[0]
            #break
        epoch_loss = running_loss/float(iters)
        epoch_g_acc = running_g_acc/float(running_edge_count)
        print("[{}] Epoch: {}/{} MTL_Loss: {:0.6f} Graph_Acc: {:0.6f}".format(\
                            'MTL-Train', epoch_count+1, epoch, epoch_loss, epoch_g_acc))
        
        checkpoint = {'state_dict': model.state_dict()}
        save_name = "checkpoints/mtl_train/"+args.mtl_version+"/checkpoint_" + str(epoch_count+1) + '_epoch.pth'
        torch.save(checkpoint, os.path.join(save_name))
        
        print("=========== Evaluation ===============")
        eval_mtl(model, dict_dataloader_val, text_field)

    return


if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
    device = torch.device('cuda')
    # arguments
    parser = argparse.ArgumentParser(description='Incremental domain adaptation for surgical report generation')
    parser.add_argument('--batch_size',            type=int,       default=8)
    parser.add_argument('--workers',               type=int,       default=0)
    parser.add_argument('--epoch',                 type=int,       default=100)
    # caption
    parser.add_argument('--exp_name',              type=str,       default='m2_transformer')
    parser.add_argument('--m',                     type=int,       default=40)   
    parser.add_argument('--cp_cbs',                type=str,       default='True')
    parser.add_argument('--cp_cbs_filter',         type=str,       default='LOG') # Potential choice: 'gau' and 'LOG'
    parser.add_argument('--cp_kernel_sizex',       type=int,       default=3)
    parser.add_argument('--cp_kernel_sizey',       type=int,       default=1)
    parser.add_argument('--cp_decay_epoch',        type=int,       default=2) 
    parser.add_argument('--cp_std_factor',         type=float,     default=0.9)
    # graph
    parser.add_argument('--gsu_cbs',               type=bool,      default=True)
    parser.add_argument('--gsu_feat',              type=str,       default='resnet18_09_SC_CBS')
    parser.add_argument('--gsu_w2v_loc',           type=str,       default='datasets/surgicalscene_word2vec.hdf5')
    # feature_extractor
    parser.add_argument('--fe_use_cbs',            type=bool,      default=True,        help='use CBS')
    parser.add_argument('--fe_std',                type=float,     default=1.0,         help='')
    parser.add_argument('--fe_std_factor',         type=float,     default=0.9,         help='')
    parser.add_argument('--fe_cbs_epoch',          type=int,       default=5,           help='')
    parser.add_argument('--fe_kernel_size',        type=int,       default=3,           help='')
    parser.add_argument('--fe_fil1',               type=str,       default='LOG',       help='gau, LOG')
    parser.add_argument('--fe_fil2',               type=str,       default='gau',       help='gau, LOG')
    parser.add_argument('--fe_fil3',               type=str,       default='gau',       help='gau, LOG')
    parser.add_argument('--fe_num_classes',        type=int,       default=11,          help='11')
    parser.add_argument('--fe_use_SC',             type=bool,      default=True,        help='use SuperCon')
    # SD file dirs
    parser.add_argument('--cp_features_path',      type=str,       default='datasets/instruments18/') 
    parser.add_argument('--cp_annotation_folder',  type=str,       default='datasets/annotations_new/annotations_SD_inc')
    parser.add_argument('--gsu_img_dir',           type=str,       default='left_frames')
    parser.add_argument('--gsu_file_dir',          type=str,       default='datasets/instruments18/')
    # TD file dirs
    # parser.add_argument('--cp_features_path',      type=str,       default='datasets/SGH_dataset_2020/') 
    # parser.add_argument('--cp_annotation_folder',  type=str,       default='datasets/annotations_new/annotations_sgh_inc')
    # parser.add_argument('--gsu_img_dir',           type=str,       default='resized_frames')
    # parser.add_argument('--gsu_file_dir',          type=str,       default='datasets/SGH_dataset_2020/')
    # checkpoints dir
    # non-common extractor, MTL, UDA, trained only on SD
    parser.add_argument('--cp_checkpoint',         type=str,       default='checkpoints/IDA_MICCAI2021_checkpoints/SD_base_LOG/')
    parser.add_argument('--gsu_checkpoint',        type=str,       default='checkpoints/g_checkpoints/da_ecbs_resnet18_09_SC_eCBS/da_ecbs_resnet18_09_SC_eCBS/epoch_train/checkpoint_D1230_epoch.pth')
    parser.add_argument('--fe_modelpath',          type=str,       default='feature_extractor/checkpoint/incremental/inc_ResNet18_SC_CBS_0_012345678.pkl')
    parser.add_argument('--mtl_version',           type=str,       default='UDA_balanced_loss')
    args = parser.parse_args()
    print(args)

    # graph scene understanding constants
    gsu_const = {}
    gsu_const['file_dir'] = args.gsu_file_dir
    gsu_const['img_dir'] = args.gsu_img_dir
    gsu_const['dataconst'] = SurgicalSceneConstants()
    gsu_const['feature_extractor'] = args.gsu_feat
    gsu_const['w2v_loc'] =args.gsu_w2v_loc


    '''==== Dataset ===='''
    # Pipeline for image regions and text
    image_field = None
    #image_field = ImageDetectionsField(detections_path=args.cp_features_path, max_detections=6, load_in_tmp=False)  
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)

    # Create the dataset 
    dataset = MTL_DATASET(image_field, text_field, gsu_const, args.cp_features_path, args.cp_annotation_folder, args.cp_annotation_folder)
    train_dataset, val_dataset = dataset.splits   
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    print('train:', len(train_dataset))
    print('val:', len(val_dataset))
    
    # Building vocabulary
    if not os.path.isfile('datasets/vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=2)  
        pickle.dump(text_field.vocab, open('datasets/vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('datasets/vocab_%s.pkl' % args.exp_name, 'rb'))

    print('vocabulary size is:', len(text_field.vocab))
    print(text_field.vocab.stoi)

    # data loader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size) # for caption with word GT class number

    '''==== MTL-model ===='''
    model = build_model(args, text_field, device)

    '''==== First evaluation MTL ===='''
    eval_mtl(model, dict_dataloader_val, text_field)

    '''==== MTL-Train model ===='''
    # train for 100 epoch
    train(args.epoch, 0.001, model, train_dataloader, dict_dataloader_val, text_field)