import os
import torch
import torch.nn as nn
import shutil
from easydict import EasyDict as edict
import pickle

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


def save_checkpoint(state, is_best, checkpoint='checkpoint',
                    filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))



def combine_dicts(feat_dict_target, feat_dict_source): # expects the easydict object
    feat_dict_combined = edict({})
    feat_dict_combined.feat_vec = torch.cat([feat_dict_target.feat_vec, feat_dict_source.feat_vec], dim=0)
    feat_dict_combined.labels = feat_dict_target.labels + feat_dict_source.labels
    feat_dict_combined.names = feat_dict_target.names + feat_dict_source.names
    feat_dict_combined.domain_identifier = feat_dict_target.domain_identifier + feat_dict_source.domain_identifier
    return feat_dict_combined

def load_bank(args,mode = 'pkl'):
    if mode == 'pkl':
        f = open("./banks/%s_unlabelled_target_%s_%s.pkl"%(args.net,args.target,args.num), "rb")
        feat_dict_target = edict(pickle.load(f))
        feat_dict_target.feat_vec = feat_dict_target.feat_vec.cuda()
        num_target = len(feat_dict_target.names)
        domain = ["T" for i in range(num_target)]
        feat_dict_target.domain_identifier = domain

        f = open("./banks/%s_labelled_source_%s.pkl"%(args.net,args.source), "rb") # Loading the feature bank for the source samples
        feat_dict_source = edict(pickle.load(f))
        feat_dict_source.feat_vec  = feat_dict_source.feat_vec.cuda() 
        num_source = len(feat_dict_source.names)
        domain = ["S" for i in range(num_source)]
        feat_dict_source.domain_identifier = domain
        # Concat the corresponsing components of the 2 dictionaries
        feat_dict_combined = edict({})
        feat_dict_combined  = combine_dicts(feat_dict_source, feat_dict_target)

    elif mode == 'random':
        feat_dict_target = edict({})
        f_target = open(os.path.join("./data/txt/%s"%(args.dataset),"unlabeled_target_images_%s_%s.txt"%(args.target,str(args.num))),"r")
        names_target, labels_target = [],[]
        for line in f_target:
            line = line.replace("\n","")
            names_target.append(line.split()[0])
            labels_target.append(int(line.split()[1]))

        feat_dict_target.names = names_target
        feat_dict_target.labels = labels_target
        feat_dim = 512 if args.net == 'resnet34' else 4096
        feat_dict_target.feat_vec = torch.randn(len(names_target),feat_dim).cuda()
        num_target = len(feat_dict_target.names)
        domain = ["T" for i in range(num_target)]
        feat_dict_target.domain_identifier = domain

        feat_dict_source = edict({})
        f_source = open(os.path.join("./data/txt/%s"%(args.dataset),"labeled_source_images_%s.txt"%(args.source)),"r")
        
        names_source,labels_source = [],[]
        for line in f_source:
            line = line.replace("\n","")
            names_source.append(line.split()[0])
            labels_source.append(int(line.split()[1]))
        feat_dict_source.feat_vec = torch.randn(len(names_source),feat_dim).cuda()
        feat_dict_source.names = names_source
        feat_dict_source.labels = labels_source
        num_source = len(feat_dict_source.names)
        domain = ["S" for i in range(num_source)]
        feat_dict_source.domain_identifier = domain
        # Concat the corresponsing components of the 2 dictionaries
        feat_dict_combined = edict({})
        feat_dict_combined  = combine_dicts(feat_dict_source, feat_dict_target)

    print("Bank keys - Target: ", feat_dict_target.keys(),"Source: ", feat_dict_source.keys())
    print("Num  - Target: ", len(feat_dict_target.names), "Source: ", len(feat_dict_source.names))

    return feat_dict_source, feat_dict_target, feat_dict_combined


def update_features(feat_dict, data, G, F1, momentum, source  = False):
    '''
    Description:
    1. This function updates the feature bank using the reprsentations of the current batch
    Inputs:
    1. feat_dict - the bank of representations taken after the image has passed through G
    2. data - the current batch
    3. G - the feature extraction vector
    4. momentum - the momentum with which we could use to update the feature bank
    5. source [bool] - whether the batch passed is of the source or not the source
    Outputs:
    1. f_batch - the reprsentations of the batch, to save further computation
    2. feat_dict - the updated feature dictionary
    '''
    names_batch = data[2]
    if not source:
        img_batch = data[0][0].cuda()
    else:
        img_batch = data[0].cuda() 
    names_batch = list(names_batch)
    idx = [feat_dict.names.index(name) for name in names_batch]
    f_batch = G(img_batch)
    #print(f_batch.shape)
    #print(len(idx))
    #print(feat_dict.feat_vec[idx])
    feat_dict.feat_vec[idx] = (momentum * feat_dict.feat_vec[idx] + (1 - momentum) * f_batch).detach()
    return f_batch, feat_dict
