import os,sys
sys.path.append("/home/lwy/data/liwuyan/01SLP_dir/text2sign")
import matplotlib.pyplot as plt
import torch
import argparse
from os.path import join as pjoin

import utils.paramUtil as paramUtil
from torch.utils.data import DataLoader
from utils.plot_script import *
from utils.get_opt import get_opt
from datasets.evaluator_models import MotionLenEstimatorBiGRU

from trainers import DDPMTrainer
from models import MotionTransformer
from utils.word_vectorizer import WordVectorizer, POS_enumerator
from utils.utils import *
from utils.motion_process import recover_from_ric


# def plot_t2m(data, result_path, caption):
#     joint = recover_from_ric(torch.from_numpy(data).float(), opt.joints_num).numpy()
#     # joint = motion_temporal_filter(joint, sigma=1)
#     plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)

import einops


def unpack_dense_skeleton(skeleton):
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


def unpack_frames(joints):
    unpacked_frames = {}
    i = 0
    while i < joints.shape[0]:
        skeleton = joints[i]
        skeleton = np.reshape(skeleton, 150)
        # print(skeleton)
        unpacked_frames[i] = unpack_dense_skeleton(skeleton)
        i += 1

    return unpacked_frames


def visulize_pose(unpacked_skeleton, context=None):
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

def plot_t2m(data, result_path, caption):

    unpacked_frames = unpack_frames(data)
    j=0

    while j < len(data):
        # pycharm
        plt.figure(dpi=600, figsize=(6, 8))
        # python
        # plt.figure(dpi=300, figsize=(100, 100))
        ax3d = plt.gca(projection='3d')
        ax3d.dist = 7
        ax3d.elev = 80
        ax3d.azim = -90
        context = visulize_pose(unpacked_frames[j])

        # ax3d.get_xaxis().set_visible(False)
        # ax3d.get_yaxis().set_visible(False)
        # ax3d.get_zaxis().set_visible(False)
        plt.axis('off')
        plt.grid(False)
        filename = result_path  + '{}_{}.png'.format('sign', j + 1)
        plt.savefig(filename)

        # plt.show()
        j = j + 1



def build_models(opt):
    encoder = MotionTransformer(
        input_feats=opt.dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, help='Opt path')
    parser.add_argument('--text', type=str, default="", help='Text description for motion generation')
    parser.add_argument('--motion_length', type=int, default=300, help='Number of frames for motion generation')
    parser.add_argument('--result_path', type=str, default="test_sample.gif", help='Path to save generation result')
    parser.add_argument('--gpu_id', type=int, default=-1, help="which gpu to use")
    args = parser.parse_args()
    
    device = torch.device('cuda:%d' % args.gpu_id if args.gpu_id != -1 else 'cpu')
    args.opt_path = "/home/lwy/data/liwuyan/01SLP_dir/text2sign/tools/checkpoints/t2m/test/opt.txt"
    opt = get_opt(args.opt_path, device)
    opt.do_denoise = True

    assert opt.dataset_name == "t2m"
    assert args.motion_length <= 300
    opt.data_root = '/home/lwy/data/liwuyan/01SLP_dir/text2sign/data/PH2014'
    opt.motion_dir = pjoin(opt.data_root, 'test_data.pkl')
    opt.text_dir = pjoin(opt.data_root, 'test_label.pkl')
    # test_split_file = pjoin(opt.data_root, 'test.txt')
    opt.max_motion_length = 300
    opt.joints_num = 50
    opt.dim_pose = 150
    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)
    # num_classes = 100 #200 // opt.unit_length

    opt.meta_dir = '/home/lwy/data/liwuyan/01SLP_dir/text2sign/tools/checkpoints/t2m/test/meta'

    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))

    encoder = build_models(opt).to(device)
    trainer = DDPMTrainer(opt, encoder)

    opt.model_dir = '/home/lwy/data/liwuyan/01SLP_dir/text2sign/tools/checkpoints/t2m/test/model/'
    trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))

    trainer.eval_mode()
    trainer.to(opt.device)
    # args.text = 'denn in westdeutschland steigen die werte auf zwanzig ja sogar auf zweiundzwanzig grad am oberrhein und nur hier an der ostsee bleibt es mit dreizehn grad kühler'

    # args.text = 'und nun die wettervorhersage für morgen sonntag den einundzwanzigsten november'
    #3
    # args.text = 'die aussichten am mittwoch im nordosten anfangs kräftiger regen sonst ist es wechselhaft im bergland schneeschauer'
    #4
    # args.text = 'und nun die wettervorhersage für morgen freitag den achtundzwanzigsten mai'
    #5
    # args.text = 'und das ist jetzt schon absehbar'
    #6
    # args.text = 'heute nacht überall frostig kalt wo es aufklart gibt es auch zweistellige minusgrade'
    #7
    # args.text = 'im nordosten fallen heute nacht einzelne schauer sonst ist es meist trocken und es klart teilweise auf'
    #8
    # args.text = 'abschied vom winter kräftiges tauwetter das wird ziemlich heftig ausfallen'
    #9
    # args.text = 'nur im nordwesten deutschlands bleibt es anfangs noch ein bisschen milder'
    #10
    args.text = 'es gibt ein paar schauer und gewitter'

    result_dict = {}
    with torch.no_grad():
        if args.motion_length != -1:
            caption = [args.text]
            m_lens = torch.LongTensor([args.motion_length]).to(device)
            pred_motions = trainer.generate(caption, m_lens, opt.dim_pose) # [T,150]
            motion = pred_motions[0].cpu().numpy() # (T,150)
            motion = motion * std + mean
            title = args.text + " #%d" % motion.shape[0]
            args.result_path = '/home/lwy/data/liwuyan/01SLP_dir/text2sign/tools/checkpoints/picture/ten25/'
            plot_t2m(motion, args.result_path, title)
