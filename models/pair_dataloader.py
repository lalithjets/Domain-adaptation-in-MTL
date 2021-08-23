import os
import json
import h5py
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader as TorchDataLoader

# caption libraries
import collections
from data.utils import nostdout
from data.example import Example


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
    def __init__(self, examples, fields, dataset_const):
        self.examples = examples
        self.fields = dict(fields)
        self.dataset_const = dataset_const
        self.file_dir = self.dataset_const['file_dir']
        self.img_dir = self.dataset_const['img_dir']
        self.dataconst = self.dataset_const['dataconst']
        self.feature_extractor = self.dataset_const['feature_extractor']
        self.word2vec = h5py.File(self.dataset_const['w2v_loc'], 'r')
        
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
            #if field_name == 'image' and field == None: 
            #    cp_data.append(np.zeros((6,512), dtype = np.float32))
            #else: 
            #    cp_data.append(field.preprocess(getattr(example, field_name)))   
            cp_data.append(np.zeros((6,512), dtype = np.float32)) if (field_name == 'image' and field == None) else cp_data.append(field.preprocess(getattr(example, field_name)))
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
        #if self.fields['image'] == None: 
        #    gsu_data['features'] = np.zeros((gsu_data['node_num'],512), dtype = np.float32)
        #else: 
        #    gsu_data['features'] = frame_data['node_features'][:]
        gsu_data['features'] = np.zeros((gsu_data['node_num'],512), dtype = np.float32) if (self.fields['image'] == None) else frame_data['node_features'][:]
        
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
            #if len(self.fields) == 1: 
            #    cp_batch_data = [cp_batch_data, ]
            #else: 
            #    cp_batch_data = list(zip(*cp_batch_data))
            cp_batch_data = [cp_batch_data, ] if len(self.fields) == 1 else list(zip(*cp_batch_data))

            for field, data in zip(self.fields.values(), cp_batch_data):
                #if field == None: 
                #    tensor = default_collate(data)
                #else: 
                #    tensor = field.process(data)
                tensor = default_collate(data) if field == None else field.process(data)
                if isinstance(tensor, collections.Sequence) and any(isinstance(t, torch.Tensor) for t in tensor):
                    tensors.extend(tensor)
                else: tensors.append(tensor)

            #if len(tensors) > 1:
            #    cp_batch_data = tensors
            #else: 
            #    cp_batch_data = tensors[0]
            cp_batch_data = tensors if len(tensors) > 1 else tensors[0]
            
            batch_data = {}
            batch_data['gsu'] = gsu_batch_data
            batch_data['cp'] = cp_batch_data
            
            return(batch_data)

        return collate


class PairedDataset(Dataset):
    def __init__(self, examples, fields, dataset_const):
        assert ('image' in fields)
        assert ('text' in fields)
        super(PairedDataset, self).__init__(examples, fields, dataset_const)
        self.image_field = self.fields['image']
        if self.image_field == None: print('no pre-extracted image featured')
        self.text_field = self.fields['text']
        
    def image_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = Dataset(self.examples, fields, self.dataset_const)
        return dataset
        

class MTL_DATASET(PairedDataset):
    def __init__(self, image_field, text_field, dataset_const, img_root, ann_root, id_root=None):
        # setting training and val root
        roots = {}
        roots['train'] = {'img': img_root, 'cap': os.path.join(ann_root, 'captions_train.json')}
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
        super(MTL_DATASET, self).__init__(examples, {'image': image_field, 'text': text_field}, dataset_const)   

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields, self.dataset_const) 
        val_split = PairedDataset(self.val_examples, self.fields, self.dataset_const)
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