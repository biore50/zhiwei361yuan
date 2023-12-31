import os,sys
sys.path.append("/home/lwy/data/liwuyan/01SLP_dir/text2sign")
from os.path import join as pjoin
import numpy as np
import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions
from utils.plot_script import *

from models import MotionTransformer
from trainers import DDPMTrainer
from datasets import Text2MotionDataset

from mmcv.runner import get_dist_info, init_dist
from mmcv.parallel import MMDistributedDataParallel
import torch
import torch.distributed as dist


def build_models(opt, dim_pose):
    encoder = MotionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder


if __name__ == '__main__':
    parser = TrainCompOptions()
    opt = parser.parse()
    rank, world_size = get_dist_info()

    opt.device = torch.device("cuda")
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if rank == 0:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.meta_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    if opt.dataset_name == 't2m':
        opt.data_root = '/home/lwy/data/liwuyan/01SLP_dir/text2sign/data/PH2014'
        opt.motion_dir = pjoin(opt.data_root, 'train_data.pkl')
        opt.text_dir = pjoin(opt.data_root, 'train_label.pkl')
        opt.joints_num = 50
        radius = 4
        fps = 30
        opt.max_motion_length = 300
        dim_pose = 150
        kinematic_chain = paramUtil.t2m_kinematic_chain
    # elif opt.dataset_name == 'kit':
    #     opt.data_root = './data/KIT-ML'
    #     opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    #     opt.text_dir = pjoin(opt.data_root, 'texts')
    #     opt.joints_num = 50 #21
    #     radius = 240 * 8
    #     fps = 12.5
    #     dim_pose = 150 #251
    #     opt.max_motion_length = 660 #196
    #     kinematic_chain = paramUtil.kit_kinematic_chain

    else:
        raise KeyError('Dataset Does Not Exist')

    dim_word = 300
    # mean_path = '/home/lwy/data/liwuyan/01SLP_dir/text2motion/data/HumanML3D/Mean.npy'
    # mean_path = '/home/lwy/data/liwuyan/01SLP_dir/text2motion/data/CSLG3D/Mean.npy'
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))
    # /home/lwy/data/liwuyan/01SLP_dir/text2sign/data/PH2014/PHOENIX-2014-T.train.corpus.csv
    train_split_file = pjoin(opt.data_root, 'PHOENIX-2014-T.train.corpus.csv')

    encoder = build_models(opt, dim_pose)
    if world_size > 1:
        encoder = MMDistributedDataParallel(
            encoder.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True)
    else:
        encoder = encoder.cuda()

    trainer = DDPMTrainer(opt, encoder)
    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file, opt.times)
    trainer.train(train_dataset)
