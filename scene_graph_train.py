'''
Paper       : Class Incremental Domain Adaptation for MTL-SD based Surgical Scene Understanding
Authors     :
Date        :
Code adopted from: 
            https://github.com/lalithjets/Learning-Domain-Generaliazation-with-Graph-Neural-Network-for-Surgical-Scene-Understanding.
'''

from __future__ import print_function

import os
import copy
import time
import argparse

import numpy as np
from PIL import Image

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import utils.io as io
from utils.vis_tool import *
from models.scene_graph import *
from models.scene_graph_dataloader import *


def seed_everything(seed=27):
    '''
    Fixing the random seeds
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def epoch_train(args, model, seq, device, dname, finetune = False):
    '''
    input:  args, model, seq, device, dname, finetune = false
    output: 
    '''    
    new_domain = False
    stop_epoch = args.epoch
    

    if finetune:
        '''
        if finetune, sample 143 from each domain and train at lower learning
        '''
        stop_epoch = args.ft_epoch
        train_dataset = SurgicalSceneDataset(seq_set = seq['train_seq'], data_dir = seq['data_dir'], \
                            img_dir = seq['img_dir'], dset = seq['dset'], dataconst = data_const, \
                            feature_extractor = args.feature_extractor, reduce_size = True)
        val_dataset = SurgicalSceneDataset(seq_set = seq['val_seq'], dset = seq['dset'], data_dir = seq['data_dir'], \
                            img_dir = seq['img_dir'], dataconst = data_const, \
                            feature_extractor = args.feature_extractor, reduce_size = False)
        dataset = {'train': train_dataset, 'val': val_dataset}
        model_old = None
    
    # train and test dataset for one domain
    elif (len(seq['train_seq']) == 1):
        '''
        if training on source domain, standard training.
        '''
        # set up dataset variable
        train_dataset = SurgicalSceneDataset(seq_set = seq['train_seq'], data_dir = seq['data_dir'], \
                            img_dir = seq['img_dir'], dset = seq['dset'], dataconst = data_const, \
                            feature_extractor = args.feature_extractor, reduce_size = False)
        val_dataset = SurgicalSceneDataset(seq_set = seq['val_seq'], data_dir = seq['data_dir'], \
                            img_dir = seq['img_dir'], dset = seq['dset'], dataconst = data_const, \
                            feature_extractor = args.feature_extractor, reduce_size = False)
        dataset = {'train': train_dataset, 'val': val_dataset}
        model_old = None
   
    # train and test for multiple domain
    elif (len(seq['train_seq']) > 1):
        '''
        if extending training to new domain, train on TD, and perform KD-based learning on sampled (143 frames)
        SD domain between the current training model and previously trained model.
        '''
        # set up dataset variable
        new_domain = True
        curr_tr_seq = seq['train_seq'][len(seq['train_seq'])-1:]
        curr_tr_data_dir = seq['data_dir'][len(seq['data_dir'])-1:]
        curr_tr_img_dir = seq['img_dir'][len(seq['img_dir'])-1:]
        curr_dset = seq['dset'][len(seq['dset'])-1:]
        #print(curr_tr_seq, curr_tr_data_dir, curr_tr_img_dir, curr_dset)
        train_dataset = SurgicalSceneDataset(seq_set = curr_tr_seq, data_dir = curr_tr_data_dir, \
                            img_dir = curr_tr_img_dir, dset = curr_dset, dataconst = data_const, \
                            feature_extractor = args.feature_extractor, reduce_size = False)
        val_dataset = SurgicalSceneDataset(seq_set = seq['val_seq'], data_dir = seq['data_dir'], \
                            img_dir = seq['img_dir'], dset = seq['dset'], dataconst = data_const, \
                            feature_extractor = args.feature_extractor, reduce_size = False)
        dataset = {'train': train_dataset, 'val': val_dataset}
        model_old = copy.deepcopy(model)
    
    # use default DataLoader() to load the data. 
    train_dataloader = DataLoader(dataset=dataset['train'], batch_size=args.batch_size, shuffle= True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=dataset['val'], batch_size=args.batch_size, shuffle= True, collate_fn=collate_fn)
    dataloader = {'train': train_dataloader, 'val': val_dataloader}
    
    # criterion and scheduler
    criterion = nn.MultiLabelSoftMarginLoss()
    # criterion = nn.BCEWithLogitsLoss()
    
    # set visualization and create folder to save checkpoints
    writer = SummaryWriter(log_dir=args.log_dir + '/' + args.exp_ver + '/' + 'epoch_train')
    io.mkdir_if_not_exists(os.path.join(args.save_dir, args.exp_ver, 'epoch_train'), recursive=True)

    for epoch in range(args.start_epoch, stop_epoch):
        
        # each epoch has a training and validation step
        epoch_acc = 0
        epoch_loss = 0
        
        # finetune
        if finetune:
            train_dataset = SurgicalSceneDataset(seq_set = seq['train_seq'], data_dir = seq['data_dir'], \
                                img_dir = seq['img_dir'], dset = seq['dset'], dataconst = data_const, \
                                feature_extractor = args.feature_extractor, reduce_size = True)
            dataset['train'] = train_dataset
            train_dataloader = DataLoader(dataset=dataset['train'], batch_size=args.batch_size, shuffle= True, collate_fn=collate_fn)
            dataloader['train'] = train_dataloader

        # build optimizer  
        if finetune: lrc = args.lr / 10
        else: lrc = args.lr
        
        if args.optim == 'sgd': 
            optimizer = optim.SGD(model.parameters(), lr= lrc, momentum=0.9, weight_decay=0)
        else: 
            optimizer = optim.Adam(model.parameters(), lr= lrc, weight_decay=0)
        
        for phase in ['train', 'val']:
            
            start_time = time.time()
            
            idx = 0
            running_acc = 0.0
            running_loss = 0.0
            running_edge_count = 0
            
            if phase == 'train' and args.use_cbs:
                model.grnn1.gnn.apply_h_h_edge.get_new_kernels(epoch)
                model.to(device)
            
            for data in dataloader[phase]:
                train_data = data
                img_name = train_data['img_name']
                img_loc = train_data['img_loc']
                node_num = train_data['node_num']
                roi_labels = train_data['roi_labels']
                det_boxes = train_data['det_boxes']
                edge_labels = train_data['edge_labels']
                edge_num = train_data['edge_num']
                features = train_data['features']
                spatial_feat = train_data['spatial_feat']
                word2vec = train_data['word2vec']
                features, spatial_feat, word2vec, edge_labels = features.to(device), spatial_feat.to(device), word2vec.to(device), edge_labels.to(device)    
                
                if phase == 'train':
                    '''
                    Standard training
                    '''
                    model.train()
                    model.zero_grad()
                    outputs = model(node_num, features, spatial_feat, word2vec, roi_labels)
                    
                    # loss and accuracy
                    if args.use_t: outputs = outputs / args.t_scale
                    loss = criterion(outputs, edge_labels.float())
                    loss.backward()
                    optimizer.step()
                    acc = np.sum(np.equal(np.argmax(outputs.cpu().data.numpy(), axis=-1), np.argmax(edge_labels.cpu().data.numpy(), axis=-1)))

                else:
                    '''
                    evaluate model
                    '''
                    model.eval()
                    # turn off the gradients for validation, save memory and computations
                    with torch.no_grad():
                        outputs = model(node_num, features, spatial_feat, word2vec, roi_labels, validation=True)
                        
                        # loss and accuracy
                        loss = criterion(outputs, edge_labels.float())
                        acc = np.sum(np.equal(np.argmax(outputs.cpu().data.numpy(), axis=-1), np.argmax(edge_labels.cpu().data.numpy(), axis=-1)))
                    
                        # print result every 1000 iteration during validation
                        if idx == 10:
                            #print(img_loc[0])
                            io.mkdir_if_not_exists(os.path.join(args.output_img_dir, ('epoch_'+str(epoch))), recursive=True)
                            image = Image.open(img_loc[0]).convert('RGB')
                            det_actions = nn.Sigmoid()(outputs[0:int(edge_num[0])])
                            det_actions = det_actions.cpu().detach().numpy()
                            action_img = vis_img(image, roi_labels[0], det_boxes[0],  det_actions, data_const, score_thresh = 0.7)
                            image = image.save(os.path.join(args.output_img_dir, ('epoch_'+str(epoch)),img_name[0]))

                idx+=1
                # accumulate loss of each batch
                running_loss += loss.item() * edge_labels.shape[0]
                running_acc += acc
                running_edge_count += edge_labels.shape[0]
            
            if phase == 'train' and new_domain:
                '''
                teacher-student Knowledge distillation based incremental domain generalization.
                '''
                # distillation loss activation
                dist_loss_act = nn.Softmax(dim=1)
                dist_loss_act = dist_loss_act.to(device)
            
                dis_seq = seq['train_seq'][:-1]
                dis_data_dir = seq['data_dir'][:-1]
                dis_img_dir = seq['img_dir'][:-1]
                dis_dset = seq['dset'][:-1]
                dis_train_dataset = SurgicalSceneDataset(seq_set =  dis_seq, data_dir = dis_data_dir, \
                                        img_dir = dis_img_dir, dset = dis_dset, dataconst = data_const, \
                                        feature_extractor = args.feature_extractor, reduce_size = True)
                dis_train_dataloader = DataLoader(dataset=dis_train_dataset, batch_size=args.batch_size, shuffle= True, collate_fn=collate_fn)
                
#                 if args.use_cbs:
#                     model_old.grnn1.gnn.apply_h_h_edge.get_new_kernels(epoch)
#                     model_old.to(device)
        
                for data in dis_train_dataloader:
                    train_data = data
                    img_name = train_data['img_name']
                    img_loc = train_data['img_loc']
                    node_num = train_data['node_num']
                    roi_labels = train_data['roi_labels']
                    det_boxes = train_data['det_boxes']
                    edge_labels = train_data['edge_labels']
                    edge_num = train_data['edge_num']
                    features = train_data['features']
                    spatial_feat = train_data['spatial_feat']
                    word2vec = train_data['word2vec']
                    features, spatial_feat, word2vec, edge_labels = features.to(device), spatial_feat.to(device), word2vec.to(device), edge_labels.to(device)    
                    
                    model.train()
                    model_old.train()
                    model.zero_grad()
                    outputs = model(node_num, features, spatial_feat, word2vec, roi_labels)
                    
                    with torch.no_grad():
                        # old network output
                        output_old = model_old(node_num, features, spatial_feat, word2vec, roi_labels)
                        output_old = Variable(output_old, requires_grad=False)
                    
                    if args.use_t:
                        outputs = outputs/args.t_scale
                        output_old = output_old/args.t_scale
                    d_loss = F.binary_cross_entropy(dist_loss_act(outputs), dist_loss_act(output_old))
                    loss = criterion(outputs, edge_labels.float()) + 0.5* d_loss
                    
                    # loss and accuracy
                    loss.backward()
                    optimizer.step()
            
            # calculate the loss and accuracy of each epoch
            epoch_loss = running_loss / len(dataset[phase])
            epoch_acc = running_acc / running_edge_count
            
            # import ipdb; ipdb.set_trace()
            # log trainval datas, and visualize them in the same graph
            if phase == 'train':
                train_loss = epoch_loss 
            else:
                writer.add_scalars('trainval_loss_epoch', {'train': train_loss, 'val': epoch_loss}, epoch)
            
            # print data
            if (epoch % args.print_every) == 0:
                end_time = time.time()
                print("[{}] Epoch: {}/{} Acc: {:0.6f} Loss: {:0.6f} Execution time: {:0.6f}".format(\
                        phase, epoch+1, args.epoch, epoch_acc, epoch_loss, (end_time-start_time)))
                        
        # scheduler.step()
        # save model
        if epoch_loss<0.0405 or epoch % args.save_every == (args.save_every - 1) and epoch >= (20-1):
            checkpoint = { 
                            'lr': args.lr,
                           'b_s': args.batch_size,
                          'bias': args.bias, 
                            'bn': args.bn, 
                       'dropout': args.drop_prob,
                        'layers': args.layers,
                    'multi_head': args.multi_attn,
                     'diff_edge': args.diff_edge,
                    'state_dict': model.state_dict()
            }
            save_name = "checkpoint_" + dname + str(epoch+1) + '_epoch.pth'
            torch.save(checkpoint, os.path.join(args.save_dir, args.exp_ver, 'epoch_train', save_name))

    writer.close()

def run_model(args, data_const):
    '''
    input : args, data_const
    '''

    # use cpu or cuda
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print('training on {}...'.format(device))

    # model
    model = AGRNN(bias=args.bias, bn=args.bn, dropout=args.drop_prob, multi_attn=args.multi_attn, layer=args.layers, diff_edge=args.diff_edge, use_cbs = args.use_cbs)
    if args.use_cbs: model.grnn1.gnn.apply_h_h_edge.get_new_kernels(0)
    
    # calculate the amount of all the learned parameters
    parameter_num = 0
    for param in model.parameters(): parameter_num += param.numel()
    print(f'The parameters number of the model is {parameter_num / 1e6} million')

    # load pretrained model
    if args.pretrained:
        print(f"loading pretrained model {args.pretrained}")
        checkpoints = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoints['state_dict'])
    model.to(device)
    
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.3) #the scheduler divides the lr by 10 every 150 epochs

    # get the configuration of the model and save some key configurations
    io.mkdir_if_not_exists(os.path.join(args.save_dir, args.exp_ver), recursive=True)
    for i in range(args.layers):
        if i==0:
            model_config = model.CONFIG1.save_config()
            model_config['lr'] = args.lr
            model_config['bs'] = args.batch_size
            model_config['layers'] = args.layers
            model_config['multi_attn'] = args.multi_attn
            model_config['data_aug'] = args.data_aug
            model_config['drop_out'] = args.drop_prob
            model_config['optimizer'] = args.optim
            model_config['diff_edge'] = args.diff_edge
            model_config['model_parameters'] = parameter_num
            io.dump_json_object(model_config, os.path.join(args.save_dir, args.exp_ver, 'l1_config.json'))
    print('save key configurations successfully...')

    # domain 1
    train_seq = [[2,3,4,6,7,9,10,11,12,14,15]]
    val_seq = [[1,5,16]]
    data_dir = ['datasets/instruments18/seq_']
    img_dir = ['/left_frames/']
    dset = [0] # 0 for ISC, 1 for SGH
    seq = {'train_seq': train_seq, 'val_seq': val_seq, 'data_dir': data_dir, 'img_dir':img_dir, 'dset': dset}
    print('======================== Domain 1 ==============================')
    epoch_train(args, model,seq, device, "D1")
    
    # Domain generalize to TD
    
    train_seq = [[2,3,4,6,7,9,10,11,12,14,15], [14,15,16,17,18,19,21,22]]
    val_seq = [[1,5,16],[1,2,3,4,5,6,7,8,9,10,11,12,13,20,23]]
    data_dir = ['datasets/instruments18/seq_', 'datasets/SGH_dataset_2020/']
    img_dir = ['/left_frames/', '/resized_frames/']
    dset = [0, 1]
    seq = {'train_seq': train_seq, 'val_seq': val_seq, 'data_dir': data_dir, 'img_dir':img_dir, 'dset': dset}
    print('======================== Domain 2 ==============================')
    epoch_train(args, model,seq, device, "D2")

    # Finetune to both SD and TD
    print('======================== Domain 1-2 FT =========================')
    epoch_train(args, model,seq, device, "D2F", finetune = True)


if __name__ == "__main__":

    # Version and feature extraction
    ver = 'da_ecbs_resnet18_11_SC_eCBS'
    f_e = 'resnet18_11_SC_CBS'

    parser = argparse.ArgumentParser(description='domain_generalization in scene understanding')
    # Hyperparams
    parser.add_argument('--lr',                type=float,   default=0.00001,                                   help='0.00001')
    parser.add_argument('--epoch',             type=int,     default=251,                                       help='251')
    parser.add_argument('--ft_epoch',          type=int,     default=81,                                        help='81')
    parser.add_argument('--start_epoch',       type=int,     default=0,                                         help='0')
    parser.add_argument('--batch_size',        type=int,     default=32,                                        help='32')
    parser.add_argument('--train_model',       type=str,     default='epoch',                                   help='epoch')
    # network
    parser.add_argument('--layers',            type=int,     default = 1,                                       help='1') 
    parser.add_argument('--bn',                type=bool,    default = False,                                   help='pass empty string for false') 
    parser.add_argument('--drop_prob',         type=float,   default = 0.3,                                     help='0.3') 
    parser.add_argument('--bias',              type=bool,    default = True,                                    help='pass empty string for false') 
    parser.add_argument('--multi_attn',        type=bool,    default = False,                                   help='pass empty string for false') 
    parser.add_argument('--diff_edge',         type=bool,    default = False,                                   help='pass empty string for false') 
    # CBS
    parser.add_argument('--use_cbs',           type=bool,     default = True,                                   help='pass empty string for false')
    # t-norm
    parser.add_argument('--use_t',             type=bool,     default = False,                                  help='pass empty string for false')
    parser.add_argument('--t_scale',           type=float,     default = 1.5,                                   help='1.5')
    # optimizer
    parser.add_argument('--optim',             type=str,     default ='adam',                                   help='sgd / adam')
    # GPU
    parser.add_argument('--gpu',               type=bool,    default=True,                                      help='pass empty string for false')
    # file locations
    parser.add_argument('--exp_ver',           type=str,     default=ver,                                       help='version_name')
    parser.add_argument('--log_dir',           type=str,     default = './log/' + ver,                          help='log_dir')
    parser.add_argument('--save_dir',          type=str,     default = './checkpoints/g_checkpoints/' + ver,    help='save_dir')
    parser.add_argument('--output_img_dir',    type=str,     default = './results/' + ver,                      help='epoch')
    parser.add_argument('--save_every',        type=int,     default = 10,                                      help='10')
    parser.add_argument('--pretrained',        type=str,     default = None,                                    help='pretrained_loc')
    # data_processing
    parser.add_argument('--sampler',           type=int,      default = 0,                                      help='0')
    parser.add_argument('--data_aug',          type=bool,     default = False,                                  help='pass empty string for false')
    parser.add_argument('--feature_extractor', type=str,      default = f_e,                                    help='feature_extractor')
    # print every
    parser.add_argument('--print_every',       type=int,     default=10,                                        help='10')
    args = parser.parse_args()

    seed_everything()
    print(args.feature_extractor)
    data_const = SurgicalSceneConstants()
    run_model(args, data_const)
