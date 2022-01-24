"""
Author: Benny
Date: Nov 2019
"""

import os
'''HYPER PARAMETER'''
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm

import config.parameter as parameter
from dataproc.xyzrot_pcd_supervised_db import SpartanSupvervisedKeypointDBConfig, SpartanSupervisedKeypointDatabase
from dataproc.supervised_keypoint_pcd_loader import SupervisedKeypointDatasetConfig, SupervisedKeypointDataset

from network.loss import RMSELoss
from models.utils import compute_rotation_matrix_from_ortho6d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    #parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_reg_seg_heatmap_msg_v2', help='model name [default: pointnet_cls]')
    parser.add_argument('--out_channel', default=9, type=int)
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()

def construct_dataset(is_train: bool) -> (torch.utils.data.Dataset, SupervisedKeypointDatasetConfig):
    # Construct the db
    db_config = SpartanSupvervisedKeypointDBConfig()
    db_config.keypoint_yaml_name = 'peg_in_hole.yaml'
    db_config.pdc_data_root = '/tmp2/r09944001/data/pdc'
    if is_train:
        db_config.config_file_path = '/tmp2/r09944001/robot-peg-in-hole-task/mankey/config/insertion_20220107.txt'
    else:
        db_config.config_file_path = '/tmp2/r90944001/robot-peg-in-hole-task/mankey/config/insertion_20220107.txt'
    database = SpartanSupervisedKeypointDatabase(db_config)

    # Construct torch dataset
    config = SupervisedKeypointDatasetConfig()
    config.network_in_patch_width = 256
    config.network_in_patch_height = 256
    config.network_out_map_width = 64
    config.network_out_map_height = 64
    config.image_database_list.append(database)
    config.is_train = is_train
    dataset = SupervisedKeypointDataset(config)
    return dataset, config

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
        

def test(model, loader, out_channel, criterion_rmse, criterion_cos):
    rot_error = []
    xyz_error = []
    heatmap_error = []
    network = model.eval()

    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points = data[parameter.pcd_key].numpy()
        points = provider.normalize_data(points)
        points = torch.Tensor(points)
        delta_rot = data[parameter.delta_rot_key]
        delta_xyz = data[parameter.delta_xyz_key]
        heatmap_target = data[parameter.heatmap_key]
        segmentation_target = data[parameter.segmentation_key]

        if not args.use_cpu:
            points = points.cuda()
            delta_rot = delta_rot.cuda()
            delta_xyz = delta_xyz.cuda()
            heatmap_target = heatmap_target.cuda()
            
        points = points.transpose(2, 1)
        heatmap_pred, action_pred = network(points)
        
        # action control
        delta_rot_pred_6d = action_pred[:, 0:6]
        delta_rot_pred = compute_rotation_matrix_from_ortho6d(delta_rot_pred_6d, args.use_cpu) # batch*3*3
        delta_xyz_pred = action_pred[:, 6:9].view(-1,3) # batch*3

        # loss computation
        loss_heatmap = criterion_rmse(heatmap_pred, heatmap_target)
        loss_r = criterion_rmse(delta_rot_pred, delta_rot)
        loss_t = (1-criterion_cos(delta_xyz_pred, delta_xyz)).mean() + criterion_rmse(delta_xyz_pred, delta_xyz)
        
        rot_error.append(loss_r.item())
        xyz_error.append(loss_t.item())
        heatmap_error.append(loss_heatmap.item())
    
    rot_error = sum(rot_error) / len(rot_error)
    xyz_error = sum(xyz_error) / len(xyz_error)
    heatmap_error = sum(heatmap_error) / len(heatmap_error)
    
    return rot_error, xyz_error, heatmap_error


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)


    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('reg_seg_heatmap_v2')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # Construct the dataset
    train_dataset, train_config = construct_dataset(is_train=True)
    # Random split
    train_set_size = int(len(train_dataset) * 0.8)
    valid_set_size = len(train_dataset) - train_set_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, valid_set_size])
    # And the dataloader
    trainDataLoader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    validDataLoader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    '''MODEL LOADING'''
    out_channel = args.out_channel
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_pointnet2_reg_seg_heatmap_v2.py', str(exp_dir))

    #network = model.get_model(out_channel, normal_channel=args.use_normals)
    network = model.get_model(out_channel)
    criterion_rmse = RMSELoss()
    criterion_cos = torch.nn.CosineSimilarity(dim=1)
    
    network.apply(inplace_relu)

    if not args.use_cpu:
        network = network.cuda()
        criterion_rmse = criterion_rmse.cuda()
        criterion_cos = criterion_cos.cuda()
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        network.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_rot_error = 99.9
    best_xyz_error = 99.9
    best_heatmap_error = 99.9

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        train_rot_error = []
        train_xyz_error = []
        train_heatmap_error = []
        network = network.train()

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            
            points =  data[parameter.pcd_key].numpy()
            points = provider.normalize_data(points)
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            heatmap_target = data[parameter.heatmap_key]
            segmentation_target = data[parameter.segmentation_key]
            #print('heatmap size', heatmap_target.size())
            #print('segmentation', segmentation_target.size())
            delta_rot = data[parameter.delta_rot_key]
            delta_xyz = data[parameter.delta_xyz_key]
            
            if not args.use_cpu:
                points = points.cuda()
                delta_rot = delta_rot.cuda()
                delta_xyz = delta_xyz.cuda()
                heatmap_target = heatmap_target.cuda()
                
            heatmap_pred, action_pred = network(points)
            # action control
            delta_rot_pred_6d = action_pred[:, 0:6]
            delta_rot_pred = compute_rotation_matrix_from_ortho6d(delta_rot_pred_6d, args.use_cpu) # batch*3*3
            delta_xyz_pred = action_pred[:, 6:9].view(-1,3) # batch*3
            
            # loss computation
            loss_heatmap = criterion_rmse(heatmap_pred, heatmap_target)
            loss_r = criterion_rmse(delta_rot_pred, delta_rot)
            loss_t = (1-criterion_cos(delta_xyz_pred, delta_xyz)).mean() + criterion_rmse(delta_xyz_pred, delta_xyz)
            
            loss = loss_r + loss_t + loss_heatmap
            loss.backward()
            optimizer.step()
            global_step += 1
            
            train_rot_error.append(loss_r.item())
            train_xyz_error.append(loss_t.item())
            train_heatmap_error.append(loss_heatmap.item())
            
        train_rot_error = sum(train_rot_error) / len(train_rot_error)
        train_xyz_error = sum(train_xyz_error) / len(train_xyz_error)
        train_heatmap_error = sum(train_heatmap_error) / len(train_heatmap_error)
        log_string('Train Rotation Error: %f' % train_rot_error)
        log_string('Train Translation Error: %f' % train_xyz_error)
        log_string('Train Heatmap Error: %f' % train_xyz_error)

        with torch.no_grad():
            rot_error, xyz_error, heatmap_error = test(network.eval(), validDataLoader, out_channel, criterion_rmse, criterion_cos)
            
            log_string('Test Rotation Error: %f, Translation Error: %f, Heatmap Error: %f' % (rot_error, xyz_error, heatmap_error))
            log_string('Best Rotation Error: %f, Translation Error: %f, Heatmap Error: %f' % (best_rot_error, best_xyz_error, heatmap_error))
            
            if (rot_error + xyz_error + heatmap_error) < (best_rot_error + best_xyz_error + best_heatmap_error):
                best_rot_error = rot_error
                best_xyz_error = xyz_error
                best_heatmap_error = heatmap_error
                best_epoch = epoch + 1
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'rot_error': rot_error,
                    'xyz_error': xyz_error,
                    'heatmap_error': heatmap_error,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
