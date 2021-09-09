'''
Paper: Task-Aware Asynchronous MTL with  Class Incremental Contrastive Learning for Surgical Scene Understanding
Lab: MMLAB
Authors: Lalithkumar Seenivasan, Mobarakol Islam, Mengya Xu, Hongliang Ren
'''

import os
import itertools
import numpy as np
import argparse, pickle

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim

# mtl model and dataloader
from models.mtl_model import *
from models.pair_dataloader import *

# single task models
from models.feature_extractor import *
from models.scene_graph import *
from data.field import TextField, RawField
from models.transformer import MemoryAugmentedEncoder_CBS
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory

# evaluation
import evaluation
from evaluation.caption_loss import *
from evaluation.scene_graph_eval import *


def seed_everything(seed=27):
    '''
    fixing random seeds to reproduce results
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(args, text_field, device, load_pretrain = True):
    '''
    Build MTL model
    1) Feature Extraction
    2) Caption Model
    3) Graph Scene Understanding Model
    '''

    '''==== Feature extractor ===='''
    # feature extraction model
    feature_encoder = SupConResNet(args=args) if args.fe_use_SC else ResNet18(args)
    if args.fe_use_cbs:
        feature_encoder.encoder.get_new_kernels(0) if args.fe_use_SC else feature_encoder.get_new_kernels(0)
        
    # based on cuda
    num_gpu = torch.cuda.device_count()
    if num_gpu > 0:
        device_ids = np.arange(num_gpu).tolist()    
        feature_encoder.encoder = nn.DataParallel(feature_encoder.encoder) if args.fe_use_SC else nn.DataParallel(feature_encoder, device_ids=device_ids)
    
    # feature extraction pre-trained weights
    feature_encoder.load_state_dict(torch.load(args.fe_modelpath))
    
    # extract the encoder layer
    if args.fe_use_SC: 
        feature_encoder = feature_encoder.encoder
    else:
        if args.fe_use_cbs: feature_encoder = nn.Sequential(*list(feature_encoder.module.children())[:-2])
        else: feature_encoder = nn.Sequential(*list(feature_encoder.module.children())[:-1])
    feature_encoder = feature_encoder.module

    ''' ==== caption model ===='''
    # caption encoder
    if args.cp_cbs == 'True':
        encoder = MemoryAugmentedEncoder_CBS(3, 0, attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': args.m})
    else: 
        encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': args.m}) 
    
    # caption decoder
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    
    # caption model
    caption_model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.cp_cbs == 'True': 
        caption_model.encoder.get_new_kernels(0, args.cp_kernel_sizex, args.cp_kernel_sizey, args.cp_decay_epoch, args.cp_std_factor, args.cp_cbs_filter)
    
    # caption load pre-trained weights
    if load_pretrain:
        pretrained_model = torch.load(args.cp_checkpoint+('%s_best.pth' % args.exp_name))
        caption_model.load_state_dict(pretrained_model['state_dict']) 

    '''==== graph model ===='''
    # graph model
    graph_su_model = AGRNN(bias= True, bn= False, dropout=0.3, multi_attn=False, layer=1, diff_edge=False, use_cbs = args.gsu_cbs)
    if args.gsu_cbs: 
        graph_su_model.grnn1.gnn.apply_h_h_edge.get_new_kernels(0)
    
    # graph load pre-trained weights
    if load_pretrain:
        pretrained_model = torch.load(args.gsu_checkpoint)
        graph_su_model.load_state_dict(pretrained_model['state_dict'])
    
    # build MTL model
    model = mtl_model(feature_encoder, graph_su_model, caption_model)
    model = model.to(device)
    return model


def eval_mtl(model, dataloader, text_field):
    '''
    Evaluate MTL model for each epoch
    Takes in model, test dataloader and text_field
    Outputs scene graph performance (Acc, mAP and recall) and 
    caption performance (BLEU and CIDEr)
    '''
    
    model.eval()

    # declare variables
    gen = {}
    gts = {}
    scores = None
    scene_graph_edge_count = 0
    scene_graph_total_acc = 0.0
    scene_graph_total_loss = 0.0
    scene_graph_logits_list = []
    scene_graph_labels_list = []
    
    # graph criterian
    scene_graph_criterion = nn.MultiLabelSoftMarginLoss()                   
    
    for it, data in tqdm(enumerate(dataloader)):
        # scene graph input
        img_loc = data['gsu']['img_loc']
        node_num = data['gsu']['node_num']
        roi_labels = data['gsu']['roi_labels']
        det_boxes = data['gsu']['det_boxes']
        edge_labels = data['gsu']['edge_labels']
        spatial_feat = data['gsu']['spatial_feat']
        word2vec = data['gsu']['word2vec']
        spatial_feat, word2vec, edge_labels = spatial_feat.to(device), word2vec.to(device), edge_labels.to(device)         
        
        # caption input
        _, caps_gt = data['cp']
        
        with torch.no_grad():
            interaction, caption = model(img_loc, det_boxes, caps_gt, node_num, spatial_feat, word2vec, roi_labels, val = True, text_field = text_field)
        
        ''' Scene graph eval '''
        # loss and accuracy
        scene_graph_loss = scene_graph_criterion(interaction, edge_labels.float())
        scene_graph_acc = np.sum(np.equal(np.argmax(interaction.cpu().data.numpy(), axis=-1), np.argmax(edge_labels.cpu().data.numpy(), axis=-1)))
        # accumulate loss and accuracy of the batch
        scene_graph_total_loss += scene_graph_loss.item() * edge_labels.shape[0]
        scene_graph_total_acc  += scene_graph_acc
        scene_graph_edge_count += edge_labels.shape[0]
        # accumulate scene graph outputs and ground truth
        scene_graph_logits_list.append(interaction)
        scene_graph_labels_list.append(edge_labels)
        
        ''' caption eval'''
        # accumulate caption outputs and ground truth
        caps_gen = text_field.decode(caption, join_words=False)
        for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
            gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
            gen['%d_%d' % (it, i)] = [gen_i, ]    
            gts['%d_%d' % (it, i)] = [gts_i,]
        
    ''' epoch performance '''
    # scene graph
    scene_graph_logits_all = torch.cat(scene_graph_logits_list).cuda()
    scene_graph_labels_all = torch.cat(scene_graph_labels_list).cuda()
    scene_graph_logits_all = F.softmax(scene_graph_logits_all, dim=1)
    scene_graph_map_value, scene_graph_recall = calibration_metrics(scene_graph_logits_all, scene_graph_labels_all)
    
    scene_graph_total_loss = scene_graph_total_loss / len(dataloader)
    scene_graph_total_acc = scene_graph_total_acc / scene_graph_edge_count

    # caption    
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)

    print('Graph : {acc: %0.4f map: %0.4f RECALL:%0.4f loss: %0.6f}' %(scene_graph_total_acc, scene_graph_map_value, scene_graph_recall, scene_graph_total_loss))
    print(print("Caption Scores :", scores))
    
    return


def train(epoch, lrc, model, dataloader, dict_dataloader_val, text_field):
    '''
    Optimization Stage 5: Fine-tune the independently trained task model with combined loss
    Takes in: Epoch no, learning rate, model, train dataloader, test dataloader and text_field.
    Outputs: Trained model.
    '''
    
    # graph criterian
    scene_graph_criterion = nn.MultiLabelSoftMarginLoss()
    
    # optimizer 
    optimizer = optim.Adam(model.parameters(), lr= lrc, weight_decay=0)
       
    for epoch_count in range(epoch):
        
        print("=========== Train ===============")
        iters = 0
        running_loss = 0.0
        running_scene_graph_acc = 0.0
        running_scene_graph_edge_count = 0

        model.train()
        
        for data in tqdm(dataloader):
            
            iters += 1
            
            # graph scene input
            img_loc = data['gsu']['img_loc']
            node_num = data['gsu']['node_num']
            roi_labels = data['gsu']['roi_labels']
            det_boxes = data['gsu']['det_boxes']
            edge_labels = data['gsu']['edge_labels']
            spatial_feat = data['gsu']['spatial_feat']
            word2vec = data['gsu']['word2vec']
            spatial_feat, word2vec, edge_labels = spatial_feat.to(device), word2vec.to(device), edge_labels.to(device)    
            
            # caption input
            caption_nodes, caps_gt = data['cp']
            caption_nodes, caps_gt = caption_nodes.to(device), caps_gt.to(device)
            
            # forward propagation
            interaction, caption = model( img_loc, det_boxes, caps_gt, node_num, spatial_feat, word2vec, roi_labels, val = False)
            
            # scene graph loss and acc calculation
            scene_graph_loss = scene_graph_criterion(interaction, edge_labels.float())
            interaction = F.softmax(interaction, dim=1)
            scene_graph_acc = np.sum(np.equal(np.argmax(interaction.cpu().data.numpy(), axis=-1), np.argmax(edge_labels.cpu().data.numpy(), axis=-1)))
                    
            # caption loss
            captain_criterion = CELossWithLS(classes=len(text_field.vocab), smoothing=0.1, gamma=0.0, isCos=False, ignore_index=text_field.vocab.stoi['<pad>'])
            caption_loss = captain_criterion(caption[:, :-1].contiguous(), caps_gt[:, 1:].contiguous())
            
            # loss calculation and back propagation
            loss = (0.5 * scene_graph_loss) + (0.5 * caption_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # accumulate batch loss and scene graph accuracy
            running_loss += loss.item()
            running_scene_graph_acc += scene_graph_acc
            running_scene_graph_edge_count += edge_labels.shape[0]
            #break
        
        # epoch loss and graphs scene acc
        epoch_loss = running_loss/float(iters)
        epoch_g_acc = running_scene_graph_acc/float(running_scene_graph_edge_count)

        print("[{}] Epoch: {}/{} MTL_Loss: {:0.6f} Graph_Acc: {:0.6f}".format('MTL-Train', epoch_count+1, epoch, epoch_loss, epoch_g_acc))
        
        # save checkpoint
        checkpoint = {'state_dict': model.state_dict()}
        save_name = "checkpoints/mtl_train/"+args.mtl_version+"/checkpoint_" + str(epoch_count+1) + '_epoch.pth'
        torch.save(checkpoint, os.path.join(save_name))
        
        # evaluate epoch performance
        print("=========== Evaluation ===============")
        eval_mtl(model, dict_dataloader_val, text_field)

    return


if __name__ == "__main__":

    domain_adaptation = 'UDA'       # UDA / DA
    current_domain = 'SD'           # SD / TD
    load_checkpoint_domain = 'SD'   # SD / TD
    
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    device = torch.device('cuda')
    
    parser = argparse.ArgumentParser(description='Class_Incremental_Contrastive_Learning_for_Surgical_Scene_Understanding')
    ''' Hyperparams'''
    parser.add_argument('--batch_size',            type=int,       default=4)
    parser.add_argument('--workers',               type=int,       default=0)
    parser.add_argument('--epoch',                 type=int,       default=100)
    parser.add_argument('--lr',                    type=float,     default=0.0000075)
    
    ''' Feature extractor '''
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
    
    ''' Caption model '''
    parser.add_argument('--exp_name',              type=str,       default='m2_transformer')
    parser.add_argument('--m',                     type=int,       default=40)   
    parser.add_argument('--cp_cbs',                type=str,       default='True')
    parser.add_argument('--cp_cbs_filter',         type=str,       default='LOG') # Potential choice: 'gau' and 'LOG'
    parser.add_argument('--cp_kernel_sizex',       type=int,       default=3)
    parser.add_argument('--cp_kernel_sizey',       type=int,       default=1)
    parser.add_argument('--cp_decay_epoch',        type=int,       default=2) 
    parser.add_argument('--cp_std_factor',         type=float,     default=0.9)
    
    ''' Scene graph''' 
    parser.add_argument('--gsu_cbs',               type=bool,      default=True)
    parser.add_argument('--gsu_feat',              type=str,       default='resnet18_09_SC_CBS')
    parser.add_argument('--gsu_w2v_loc',           type=str,       default='datasets/surgicalscene_word2vec.hdf5')
    
    ''' Unsupervised domain adaptation '''
    if domain_adaptation == 'UDA':
        if current_domain == 'SD':
            # #Source domain file dirs
            parser.add_argument('--cp_features_path',      type=str,       default='datasets/instruments18/') 
            parser.add_argument('--cp_annotation_folder',  type=str,       default='datasets/caption_annotations_SC_CBS/annotations_SD_base')
            parser.add_argument('--gsu_img_dir',           type=str,       default='left_frames')
            parser.add_argument('--gsu_file_dir',          type=str,       default='datasets/instruments18/')
        elif current_domain == 'TD':
            # #Target domain file dirs
            parser.add_argument('--cp_features_path',      type=str,       default='datasets/SGH_dataset_2020/') 
            parser.add_argument('--cp_annotation_folder',  type=str,       default='datasets/caption_annotations_SC_CBS/annotations_TD_base')
            parser.add_argument('--gsu_img_dir',           type=str,       default='resized_frames')
            parser.add_argument('--gsu_file_dir',          type=str,       default='datasets/SGH_dataset_2020/')
        # UDA pre-trained checkpoint weights
        parser.add_argument('--cp_checkpoint',         type=str,       default='checkpoints/c_checkpoints/SD_base_LOG/')
        parser.add_argument('--gsu_checkpoint',        type=str,       default='checkpoints/g_checkpoints/da_ecbs_resnet18_09_SC_eCBS/da_ecbs_resnet18_09_SC_eCBS/epoch_train/checkpoint_D1230_epoch.pth')
        parser.add_argument('--fe_modelpath',          type=str,       default='feature_extractor/checkpoint/incremental/inc_ResNet18_SC_CBS_0_012345678.pkl')

    ''' Incremental domain adaptation '''
    if domain_adaptation == 'DA':
        if current_domain == 'SD':
            # Source domain incremental file dirs
            parser.add_argument('--cp_features_path',      type=str,       default='datasets/instruments18/') 
            parser.add_argument('--cp_annotation_folder',  type=str,       default='datasets/caption_annotations_SC_CBS/annotations_SD_inc')
            parser.add_argument('--gsu_img_dir',           type=str,       default='left_frames')
            parser.add_argument('--gsu_file_dir',          type=str,       default='datasets/instruments18/')
        elif current_domain == 'TD':
            # Target domain incremental file dirs
            parser.add_argument('--cp_features_path',      type=str,       default='datasets/SGH_dataset_2020/') 
            parser.add_argument('--cp_annotation_folder',  type=str,       default='datasets/caption_annotations_SC_CBS/annotations_TD_inc')
            parser.add_argument('--gsu_img_dir',           type=str,       default='resized_frames')
            parser.add_argument('--gsu_file_dir',          type=str,       default='datasets/SGH_dataset_2020/')
        if load_checkpoint_domain == 'SD':
            # Source domain incremental domain adaption pre-trained checkpoint weights    
            parser.add_argument('--cp_checkpoint',         type=str,       default='checkpoints/c_checkpoints/SD_inc_LOG/')
            parser.add_argument('--gsu_checkpoint',        type=str,       default='checkpoints/g_checkpoints/da_ecbs_resnet18_11_SC_eCBS/da_ecbs_resnet18_11_SC_eCBS/epoch_train/checkpoint_D1250_epoch.pth')
            parser.add_argument('--fe_modelpath',          type=str,       default='feature_extractor/checkpoint/incremental/inc_ResNet18_SC_CBS_0_012345678910.pkl')
        elif load_checkpoint_domain == 'TD':
            # Target domain incremental domain adaption pre-trained checkpoint weights    
            parser.add_argument('--cp_checkpoint',         type=str,       default='checkpoints/c_checkpoints/few_shot_TD_inc_LOG/')
            parser.add_argument('--gsu_checkpoint',        type=str,       default='checkpoints/g_checkpoints/da_ecbs_resnet18_11_SC_eCBS/da_ecbs_resnet18_11_SC_eCBS/epoch_train/checkpoint_D2210_epoch.pth')
            parser.add_argument('--fe_modelpath',          type=str,       default='feature_extractor/checkpoint/incremental/inc_ResNet18_SC_CBS_0_012345678910.pkl')

    parser.add_argument('--mtl_version',           type=str,       default='MTL_FT_DA_SC_CBS')
    args = parser.parse_args()
    print(args)

    # seed randoms
    seed_everything()
    
    '''======================================= Dataset ======================================='''
    # Scene graph constants
    gsu_const = {}
    gsu_const['file_dir'] = args.gsu_file_dir
    gsu_const['img_dir'] = args.gsu_img_dir
    gsu_const['dataconst'] = SurgicalSceneConstants()
    gsu_const['feature_extractor'] = args.gsu_feat
    gsu_const['w2v_loc'] =args.gsu_w2v_loc

    # caption constants
    image_field = None # image field is set to none to enable feature extraction using feature extraction layer.
    # image_field = ImageDetectionsField(detections_path=args.cp_features_path, max_detections=6, load_in_tmp=False)  
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

    # train and validation dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size, shuffle = False) # for caption with word GT class number

    '''==================================== Build MTL-model ======================================'''
    '''
    Algorithm 1
    Load pretrained weights of MTL sub-modules
    1) Feature extrator module: pre-trained weights from optimization Stage 1
    2) Scene graph module: pre-trained weights from optimization from stage 3
    3) Caption module: pre=trained weights from optimization from stage 4
    '''
    model = build_model(args, text_field, device, load_pretrain=False)
    
    '''================================= initial model evaluation ================================'''
    eval_mtl(model, dict_dataloader_val, text_field)

    '''Algorithm 1, Optimization Stage 5: Fine-tune the independently trained task model with combined loss'''
    # print('Learning_rate: ', args.lr)
    train(args.epoch, args.lr, model, train_dataloader, dict_dataloader_val, text_field)