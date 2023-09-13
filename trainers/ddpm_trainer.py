import torch
import torch.nn.functional as F
import random
import time
import sys, os
from torch.utils.tensorboard import SummaryWriter
from models.transformer import MotionTransformer
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
from utils.utils import print_current_loss
from os.path import join as pjoin
import codecs as cs
import torch.distributed as dist
from IPython import embed
import einops
import numpy as np
import matplotlib.pyplot as plt

from mmcv.runner import get_dist_info
from models.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)

from datasets import build_dataloader


class DDPMTrainer(object):

    def __init__(self, args, encoder):
        self.opt = args
        self.device = args.device
        self.encoder = encoder
        self.diffusion_steps = args.diffusion_steps
        sampler = 'uniform'
        beta_scheduler = 'linear'
        betas = get_named_beta_schedule(beta_scheduler, self.diffusion_steps)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE
        )
        self.sampler = create_named_schedule_sampler(sampler, self.diffusion)
        self.sampler_name = sampler
        self.train_writer = SummaryWriter(os.path.join('tensorboard', 'train'), 'train')

        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.to(self.device)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data, eval_mode=False):
        caption, cap_token, c_length, gloss, gloss_token, g_length, motions, m_lens = batch_data
        # caption, motions, m_lens = batch_data
        motions = motions.detach().to(self.device).float()


        # print(caption)
        # meta_dir = '/home/lwy/data/liwuyan/01SLP_dir/text2sign/tools/checkpoints/t2m/test/meta'
        #
        # mean = np.load(pjoin(meta_dir, 'mean.npy'))
        # std = np.load(pjoin(meta_dir, 'std.npy'))
        #
        # motion = motions[22].cpu().numpy()  # (T,150)
        # motion = motion * std + mean
        # title = "this dog" + " #%d" % motion.shape[0]
        # result_path = '/home/lwy/data/liwuyan/01SLP_dir/text2sign/tools/checkpoints/picture/23/'
        # self.plot_t2m(motion, result_path, title)

        self.caption = caption
        # embed()
        # print('cap_token=', cap_token)
        self.motions = motions
        x_start = motions
        B, T = x_start.shape[:2]
        cur_len = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(self.device)
        t, _ = self.sampler.sample(B, x_start.device)
        output = self.diffusion.training_losses(
            model=self.encoder,
            x_start=x_start,
            t=t,
            model_kwargs={"text": caption, "length": cur_len}
        )

        self.real_noise = output['target']
        self.fake_noise = output['pred']
        try:
            self.src_mask = self.encoder.module.generate_src_mask(T, cur_len).to(x_start.device)
        except:
            self.src_mask = self.encoder.generate_src_mask(T, cur_len).to(x_start.device)

    def plot_t2m(self, data, result_path, caption):

        unpacked_frames = self.unpack_frames(data)
        j = 0

        while j < len(data):
            # pycharm
            plt.figure(dpi=600, figsize=(6, 8))
            # python
            # plt.figure(dpi=300, figsize=(100, 100))
            ax3d = plt.gca(projection='3d')
            ax3d.dist = 7
            ax3d.elev = 80
            ax3d.azim = -90
            context = self.visulize_pose(unpacked_frames[j])

            # ax3d.get_xaxis().set_visible(False)
            # ax3d.get_yaxis().set_visible(False)
            # ax3d.get_zaxis().set_visible(False)
            plt.axis('off')
            plt.grid(False)
            filename = result_path + '{}_{}.png'.format('sign', j + 1)
            plt.savefig(filename)

            # plt.show()
            j = j + 1

    def visulize_pose(self, unpacked_skeleton, context=None):
        POSE_LINKS = [(4, 3), (3, 2), (7, 6), (6, 5), (5, 1), (2, 1),
                      (0, 1)]

        HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4),
                      (0, 5), (5, 6), (6, 7), (7, 8),
                      (0, 9), (9, 10), (10, 11), (11, 12),
                      (0, 13), (13, 14), (14, 15), (15, 16),
                      (0, 17), (17, 18), (18, 19), (19, 20)]

        LINKS = {
            'pose': POSE_LINKS, 'left_hand': HAND_LINKS, 'right_hand': HAND_LINKS
        }

        def add_points(coords, context):
            if context is not None:
                coords[np.asanyarray(context['masked']), :] = 0
            coords = coords[coords[:, 0] != 0]
            plt.scatter(coords[:, 0], coords[:, 1], s=0.2)

        def add_link(coord_1, coord_2, context):
            # print(*list(zip(coord_1,coord_2)))
            if not (1 in coord_1 or 1 in coord_2):
                plt.plot(*list(zip(coord_1, coord_2)))

        for k, points in unpacked_skeleton.items():  # (18,2)
            points = points.copy()
            # if k == 'pose':
            #     context = {'masked': [8, 11, 9, 12, 10, 13]}
            # else:
            #     context = None
            points[:, 1] = 1 - points[:, 1]

            for (p1, p2) in LINKS[k]:
                add_link(points[p1], points[p2], context)
            add_points(points, context)
        return context

    def unpack_frames(self, joints):
        unpacked_frames = {}
        i = 0
        while i < joints.shape[0]:
            skeleton = joints[i]
            skeleton = np.reshape(skeleton, 150)
            # print(skeleton)
            unpacked_frames[i] = self.unpack_dense_skeleton(skeleton)
            i += 1

        return unpacked_frames

    def unpack_dense_skeleton(self, skeleton):
        assert len(skeleton) == 150, f'应该有150个点，但是有{len(skeleton)},{skeleton}'
        # pose,face,left_hand,right_hand
        num_points = [8, 21, 21]
        part_name = ['pose', 'left_hand', 'right_hand']
        s, e = 0, 0
        parts_dict = {}
        for n, name in zip(num_points, part_name):
            e = s + 3 * n
            part_points = np.asarray(skeleton[s:e])
            part_points = einops.rearrange(part_points, '(n w)->n w', w=3)
            s = e
            parts_dict[name] = part_points
        return parts_dict


    def generate_batch(self, caption, m_lens, dim_pose):
        xf_proj, xf_out = self.encoder.encode_text(caption, self.device)
        
        B = len(caption)
        T = min(m_lens.max(), self.encoder.num_frames)
        output = self.diffusion.p_sample_loop(
            self.encoder,
            (B, T, dim_pose),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                'xf_proj': xf_proj,
                'xf_out': xf_out,
                'length': m_lens
            })
        return output

    def generate(self, caption, m_lens, dim_pose, batch_size=1024):
        N = len(caption)
        cur_idx = 0
        self.encoder.eval()
        all_output = []
        while cur_idx < N:
            if cur_idx + batch_size >= N:
                batch_caption = caption[cur_idx:]
                batch_m_lens = m_lens[cur_idx:]
            else:
                batch_caption = caption[cur_idx: cur_idx + batch_size]
                batch_m_lens = m_lens[cur_idx: cur_idx + batch_size]
            output = self.generate_batch(batch_caption, batch_m_lens, dim_pose)
            B = output.shape[0]

            for i in range(B):
                all_output.append(output[i])
            cur_idx += batch_size
        return all_output

    def backward_G(self):
        loss_mot_rec = self.mse_criterion(self.fake_noise, self.real_noise).mean(dim=-1)
        loss_mot_rec = (loss_mot_rec * self.src_mask).sum() / self.src_mask.sum()
        self.loss_mot_rec = loss_mot_rec
        loss_logs = OrderedDict({})
        loss_logs['loss_mot_rec'] = self.loss_mot_rec.item()
        return loss_logs

    def update(self):
        self.zero_grad([self.opt_encoder])
        loss_logs = self.backward_G()
        self.loss_mot_rec.backward()
        self.clip_norm([self.encoder])
        self.step([self.opt_encoder])

        return loss_logs

    def to(self, device):
        if self.opt.is_train:
            self.mse_criterion.to(device)
        self.encoder = self.encoder.to(device)

    def train_mode(self):
        self.encoder.train()

    def eval_mode(self):
        self.encoder.eval()

    def save(self, file_name, ep, total_it):
        state = {
            'opt_encoder': self.opt_encoder.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        try:
            state['encoder'] = self.encoder.module.state_dict()
        except:
            state['encoder'] = self.encoder.state_dict()
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        if self.opt.is_train:
            self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
        self.encoder.load_state_dict(checkpoint['encoder'], strict=True)
        return checkpoint['ep'], checkpoint.get('total_it', 0)

    def train(self, train_dataset):
        rank, world_size = get_dist_info()
        self.to(self.device)
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.opt.lr)
        it = 0
        cur_epoch = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            cur_epoch, it = self.load(model_dir)

        start_time = time.time()

        train_loader = build_dataloader(
            train_dataset,
            samples_per_gpu=self.opt.batch_size,
            drop_last=True,
            workers_per_gpu=4,
            shuffle=True)

        logs = OrderedDict()
        for epoch in range(cur_epoch, self.opt.num_epochs):
            self.train_mode()
            flag = True
            for i, batch_data in enumerate(train_loader):
                self.forward(batch_data)
                log_dict = self.update()
                if flag:
                    self.train_writer.add_scalar('tra_loss', self.loss_mot_rec, epoch)
                    print('tran-loss', self.loss_mot_rec, epoch)
                    flag= False

                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v
                it += 1
                if it % self.opt.log_every == 0 and rank == 0:
                    mean_loss = OrderedDict({})
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, mean_loss, epoch, inner_iter=i)



                if it % self.opt.save_latest == 0 and rank == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if rank == 0:
                self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if epoch % self.opt.save_every_e == 0 and rank == 0:
                self.save(pjoin(self.opt.model_dir, 'ckpt_e%03d.tar'%(epoch)),
                            epoch, total_it=it)
