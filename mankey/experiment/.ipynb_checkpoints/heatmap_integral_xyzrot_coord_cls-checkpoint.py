import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import random
from torch.utils.data import DataLoader
import time
import sys
sys.path.append('/tmp2/r09944001/robot-peg-in-hole-task')

from mankey.network.resnet_nostage_xyzrot_cls import ResnetNoStageConfig, ResnetNoStage, init_from_modelzoo
from mankey.network.weighted_loss import weighted_mse_loss, weighted_l1_loss
import mankey.network.predict as predict
import mankey.network.visualize_dbg as visualize_dbg
import mankey.config.parameter as parameter
from mankey.dataproc.xyzrot_supervised_db import SpartanSupvervisedKeypointDBConfig, SpartanSupervisedKeypointDatabase
from mankey.dataproc.supervised_keypoint_loader import SupervisedKeypointDatasetConfig, SupervisedKeypointDataset
from torch.utils.tensorboard import SummaryWriter

#from mankey.network.control_network import ControlNetwork
from mankey.network.loss import RMSELoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter()
enableKeypointPos = False
#checkpoint_dir_name = 'ckpnt_xyzrot_coord_small_range_1015_kpt_cls_filter_400'
checkpoint_dir_name = 'ckpnt_xyzrot_coord_small_range_1015_no_kpt_cls_5_25_25_filter_400'

# Some global parameter
learning_rate = 2e-4
n_epoch = 201
def construct_dataset(is_train: bool) -> (torch.utils.data.Dataset, SupervisedKeypointDatasetConfig):
    # Construct the db
    db_config = SpartanSupvervisedKeypointDBConfig()
    #db_config.keypoint_yaml_name = 'mug_3_keypoint_image.yaml'
    db_config.keypoint_yaml_name = 'peg_in_hole_filter_400_5_25_25.yaml'
    db_config.pdc_data_root = '/tmp2/r09944001/data/pdc'
    if is_train:
        db_config.config_file_path = '/tmp2/r09944001/robot-peg-in-hole-task/mankey/config/insertion_20211015.txt'
    else:
        db_config.config_file_path = '/tmp2/r90944001/robot-peg-in-hole-task/mankey/config/insertion_20211015.txt'
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


def construct_network():
    net_config = ResnetNoStageConfig()
    net_config.num_keypoints = 4
    net_config.image_channels = 4
    net_config.depth_per_keypoint = 2   # depthmap_pred set 2 -> (:,3,:,:)  set 3 -> (:,6,:,:)
    net_config.num_layers = 34
    network = ResnetNoStage(net_config)
    return network, net_config


def visualize(network_path: str, save_dir: str):
    # Get the network
    network, _ = construct_network()

    # Load the network
    network.load_state_dict(torch.load(network_path))
    network.to(device)
    network.eval()

    # Construct the dataset
    dataset, config = construct_dataset(is_train=False)

    # try the entry
    num_entry = 50
    entry_idx = []
    for i in range(num_entry):
        entry_idx.append(random.randint(0, len(dataset) - 1))

    # A good example and a bad one
    for i in range(len(entry_idx)):
        visualize_dbg.visualize_entry_nostage(entry_idx[i], network, dataset, config, save_dir)


def train(checkpoint_dir: str, start_from_ckpnt: str = '', save_epoch_offset: int = 0):
    # Construct the dataset
    dataset_train, train_config = construct_dataset(is_train=True)
    # dataset_val, val_config = construct_dataset(is_train=False)

    # Random split
    train_set_size = int(len(dataset_train) * 0.8)
    valid_set_size = len(dataset_train) - train_set_size
    dataset_train, dataset_valid = torch.utils.data.random_split(dataset_train, [train_set_size, valid_set_size])


    # And the dataloader
    loader_train = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True, num_workers=4)
    loader_valid = DataLoader(dataset=dataset_valid, batch_size=32, shuffle=False, num_workers=4)

    # Construct the regressor
    network, net_config = construct_network()
    #control_network = ControlNetwork(in_channel=int(net_config.num_keypoints * net_config.depth_per_keypoint * 256/4 * 256/4))
    if len(start_from_ckpnt) > 0:
        network.load_state_dict(torch.load(start_from_ckpnt))
    else:
        init_from_modelzoo(network, net_config)
    network.to(device)
    #control_network.to(device)

    # The checkpoint
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    # root mean square error loss
    criterion_rmse = RMSELoss()
    criterion_cos = torch.nn.CosineSimilarity(dim=1)
    criterion_bce = torch.nn.BCELoss(reduction='none')
    criterion_mse = torch.nn.MSELoss()
    criterion_ce = torch.nn.CrossEntropyLoss()
    # The optimizer and scheduler
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 90], gamma=0.1)
    

    # The training loop
    for epoch in range(n_epoch):
        # Save the network
        if epoch % 20 == 0 and epoch > 0:
            file_name = 'checkpoint-%d.pth' % (epoch + save_epoch_offset)
            checkpoint_path = os.path.join(checkpoint_dir, file_name)
            print('Save the network at %s' % checkpoint_path)
            torch.save(network.state_dict(), checkpoint_path)

        # Prepare info for training
        network.train()
        train_error_xy = 0
        train_error_depth = 0
        train_error_move = 0
        train_error_rot = 0
        train_error_xyz = 0
        train_error_step = 0
        train_rot_x_acc = []
        train_rot_y_acc = []
        train_rot_z_acc = []

        # The learning rate step
        scheduler.step()
        for param_group in optimizer.param_groups:
            print('The learning rate is ', param_group['lr'])

        # The training iteration over the dataset
        for idx, data in enumerate(loader_train):
            # Get the data
            image = data[parameter.rgbd_image_key]
            keypoint_xy_depth = data[parameter.keypoint_xyd_key]
            keypoint_weight = data[parameter.keypoint_validity_key]
            delta_rot_cls = data[parameter.delta_rot_cls_key] 
            delta_xyz = data[parameter.delta_xyz_key]
            gripper_pose = data[parameter.gripper_pose_key]
            step_size = data[parameter.step_size_key]

            # Upload to GPU
            image = image.to(device)
            keypoint_xy_depth = keypoint_xy_depth.to(device)
            keypoint_weight = keypoint_weight.to(device)
            rot_x = delta_rot_cls[:, 2].to(device)
            rot_y = delta_rot_cls[:, 1].to(device)
            rot_z = delta_rot_cls[:, 0].to(device)
            delta_xyz = delta_xyz.to(device)
            gripper_pose = gripper_pose.to(device)
            step_size = step_size.to(device)
            #print('delta_rot',delta_rot.shape)
            #print('delta_xyz',delta_xyz.shape)
            #print('gripper_pose',gripper_pose.shape)
            #print('step_size',step_size.shape)
            # To predict
            optimizer.zero_grad()
            
            # raw_pred (batch_size, num_keypoint*2, network_out_map_height, network_out_map_width)
            # prob_pred (batch_size, num_keypoint, network_out_map_height, network_out_map_width)
            # depthmap_pred (batch_size, num_keypoint, network_out_map_height, network_out_map_width)
            xy_depth_pred, delta_rot_x_pred, delta_rot_y_pred, delta_rot_z_pred, delta_xyz_pred, step_size_pred = network(image, gripper_pose, device, enableKeypointPos=enableKeypointPos)
            #print((1-criterion_cos(torch.tensor([[0.01,0.01,0.01],[0.01,0.01,0.01]]).to(device), torch.tensor([[0.0,0.0,0.0],[0.0,0.0,0.0]]).to(device))).mean())
            #gripper control network
            #raw_pred_flatten = torch.flatten(raw_pred, start_dim=1)
            #delta_rot_pred, delta_xyz_pred, step_size_pred = control_network(raw_pred_flatten)
            
            #identity = torch.eye(3).unsqueeze(0).repeat(image.shape[0],1,1).to(device)
            #identity_hat = torch.matmul(torch.transpose(delta_rot_pred, 1, 2), delta_rot) 
            #loss_r = criterion_mse(identity, identity_hat)
            #loss_r = criterion_rmse(delta_rot_pred, delta_rot)
            loss_r_x = criterion_ce(delta_rot_x_pred, rot_x)
            loss_r_y = criterion_ce(delta_rot_y_pred, rot_y)
            loss_r_z = criterion_ce(delta_rot_z_pred, rot_z)
            loss_t = (1-criterion_cos(delta_xyz_pred, delta_xyz)).mean() + criterion_rmse(delta_xyz_pred, delta_xyz)
            #loss_t = criterion_rmse(delta_xyz_pred, delta_xyz)
            loss_s = criterion_bce(step_size_pred, step_size)
            loss_s = loss_s.mean()

            ''' 
            prob_pred = raw_pred[:, 0:net_config.num_keypoints, :, :]
            depthmap_pred = raw_pred[:, net_config.num_keypoints:, :, :]
            # heatmap (batch_size, num_keypoint, network_out_map_height, network_out_map_width)
            heatmap = predict.heatmap_from_predict(prob_pred, net_config.num_keypoints)
            _, _, heatmap_height, heatmap_width = heatmap.shape
            #print(raw_pred.shape)
            #print(prob_pred.shape)
            #print(depthmap_pred.shape)
            # Compute the coordinate
            if device == 'cpu':
                coord_x, coord_y = predict.heatmap2d_to_normalized_imgcoord_cpu(heatmap, net_config.num_keypoints)
            else:
                coord_x, coord_y = predict.heatmap2d_to_normalized_imgcoord_gpu(heatmap, net_config.num_keypoints)
            depth_pred = predict.depth_integration(heatmap, depthmap_pred)
            # Concantate them
            xy_depth_pred = torch.cat((coord_x, coord_y, depth_pred), dim=2)
            '''
            # Compute loss
            loss_kpt = weighted_l1_loss(xy_depth_pred, keypoint_xy_depth, keypoint_weight)
            #loss =loss_kpt*15 + loss_r*10 + loss_t*10 + loss_s
            loss_r = loss_r_x + loss_r_y + loss_r_z
            #loss = loss_kpt + loss_r + loss_t + loss_s
            if enableKeypointPos:
                if epoch < 100:
                    loss = loss_kpt
                else:
                    loss = loss_kpt*100 + loss_r + loss_t*10
            else:
                loss = loss_r + loss_t*10
            loss.backward()
            optimizer.step()

            rot_x_acc = (delta_rot_x_pred.argmax(dim=-1) == rot_x).float().mean()
            rot_y_acc = (delta_rot_y_pred.argmax(dim=-1) == rot_y).float().mean()
            rot_z_acc = (delta_rot_z_pred.argmax(dim=-1) == rot_z).float().mean()
            
            # Log info
            xy_error = float(weighted_l1_loss(xy_depth_pred[:, :, 0:2], keypoint_xy_depth[:, :, 0:2], keypoint_weight[:, :, 0:2]).item())
            depth_error = float(weighted_l1_loss(xy_depth_pred[:, :, 2], keypoint_xy_depth[:, :, 2], keypoint_weight[:, :, 2]).item()) 
            '''
            if idx % 100 == 0:
                print('Iteration %d in epoch %d' % (idx, epoch))
                print('The averaged pixel error is (pixel in 256x256 image): ', 256 * xy_error / len(xy_depth_pred))
                print('The averaged depth error is (mm): ', 256 * depth_error / len(xy_depth_pred))
                print('The move error is', loss_move.item())
            '''
            # Update info
            train_error_xy += float(xy_error)
            train_error_depth += float(depth_error)
            train_error_rot += float(loss_r)
            train_error_xyz += float(loss_t)
            train_error_step += float(loss_s)
            train_rot_x_acc.append(rot_x_acc)
            train_rot_y_acc.append(rot_y_acc)
            train_rot_z_acc.append(rot_z_acc)
            # cleanup
            del loss
        
        valid_error_xy = 0
        valid_error_depth = 0
        valid_error_rot = 0
        valid_error_xyz = 0
        valid_error_step = 0
        valid_rot_x_acc = []
        valid_rot_y_acc = []
        valid_rot_z_acc = []
            
        for idx, data in enumerate(loader_valid):
            # Get the data
            image = data[parameter.rgbd_image_key]
            keypoint_xy_depth = data[parameter.keypoint_xyd_key]
            keypoint_weight = data[parameter.keypoint_validity_key]
            delta_rot_cls = data[parameter.delta_rot_cls_key] 
            delta_xyz = data[parameter.delta_xyz_key]
            gripper_pose = data[parameter.gripper_pose_key]
            step_size = data[parameter.step_size_key]

            # Upload to GPU
            image = image.to(device)
            keypoint_xy_depth = keypoint_xy_depth.to(device)
            keypoint_weight = keypoint_weight.to(device)
            rot_x = delta_rot_cls[:, 2].to(device)
            rot_y = delta_rot_cls[:, 1].to(device)
            rot_z = delta_rot_cls[:, 0].to(device)
            delta_xyz = delta_xyz.to(device)
            gripper_pose = gripper_pose.to(device)
            step_size = step_size.to(device)

            with torch.no_grad():
                xy_depth_pred, delta_rot_x_pred, delta_rot_y_pred, delta_rot_z_pred, delta_xyz_pred, step_size_pred = network(image, gripper_pose, device, enableKeypointPos=enableKeypointPos)
            
            #Compute loss
            loss_r_x = criterion_ce(delta_rot_x_pred, rot_x)
            loss_r_y = criterion_ce(delta_rot_y_pred, rot_y)
            loss_r_z = criterion_ce(delta_rot_z_pred, rot_z)
            loss_t = (1-criterion_cos(delta_xyz_pred, delta_xyz)).mean() + criterion_rmse(delta_xyz_pred, delta_xyz)
            loss_s = criterion_bce(step_size_pred, step_size)
            loss_s = loss_s.mean()
            loss_kpt = weighted_l1_loss(xy_depth_pred, keypoint_xy_depth, keypoint_weight)
            loss_r = loss_r_x + loss_r_y + loss_r_z
            
            rot_x_acc = (delta_rot_x_pred.argmax(dim=-1) == rot_x).float().mean()
            rot_y_acc = (delta_rot_y_pred.argmax(dim=-1) == rot_y).float().mean()
            rot_z_acc = (delta_rot_z_pred.argmax(dim=-1) == rot_z).float().mean()
            
            xy_error = float(weighted_l1_loss(xy_depth_pred[:, :, 0:2], keypoint_xy_depth[:, :, 0:2], keypoint_weight[:, :, 0:2]).item())
            depth_error = float(weighted_l1_loss(xy_depth_pred[:, :, 2], keypoint_xy_depth[:, :, 2], keypoint_weight[:, :, 2]).item()) 
            
            valid_error_xy += float(xy_error)
            valid_error_depth += float(depth_error)
            valid_error_rot += float(loss_r)
            valid_error_xyz += float(loss_t)
            valid_error_step += float(loss_s)
            valid_rot_x_acc.append(rot_x_acc)
            valid_rot_y_acc.append(rot_y_acc)
            valid_rot_z_acc.append(rot_z_acc)
       
        # The info at epoch level
        print('Epoch %d' % epoch)
        print('The training averaged pixel error is (pixel in 256x256 image): ', 256 * train_error_xy / len(dataset_train))
        print('The training averaged depth error is (mm): ', train_config.depth_image_scale * train_error_depth / len(dataset_train))
        print('The training averaged rot error is: ', train_error_rot / len(dataset_train))
        print('The training averaged xyz error is: ', train_error_xyz / len(dataset_train))
        print('The training averaged rot_x acc is: ', sum(train_rot_x_acc) / len(train_rot_x_acc))
        print('The training averaged rot_y acc is: ', sum(train_rot_y_acc) / len(train_rot_y_acc))
        print('The training averaged rot_z acc is: ', sum(train_rot_z_acc) / len(train_rot_z_acc))
        #print('The training averaged step error is: ', train_error_step / len(dataset_train))
        
        print('The valid averaged pixel error is (pixel in 256x256 image): ', 256 * valid_error_xy / len(dataset_valid))
        print('The valid averaged depth error is (mm): ', train_config.depth_image_scale * valid_error_depth / len(dataset_valid))
        print('The valid averaged rot error is: ', valid_error_rot / len(dataset_valid))
        print('The valid averaged xyz error is: ', valid_error_xyz / len(dataset_valid))
        print('The valid averaged rot_x acc is: ', sum(valid_rot_x_acc) / len(valid_rot_x_acc))
        print('The valid averaged rot_y acc is: ', sum(valid_rot_y_acc) / len(valid_rot_y_acc))
        print('The valid averaged rot_z acc is: ', sum(valid_rot_z_acc) / len(valid_rot_z_acc))
        
        writer.add_scalar('train average pixel error', 256 * train_error_xy / len(dataset_train) , epoch)
        writer.add_scalar('train average depth error', train_config.depth_image_scale * train_error_depth / len(dataset_train) , epoch)
        writer.add_scalar('train average move error', train_error_move / len(dataset_train) , epoch)
        writer.add_scalar('trainaverage rot error', train_error_rot / len(dataset_train) , epoch)
        writer.add_scalar('train average xyz error', train_error_xyz / len(dataset_train) , epoch)
        writer.add_scalar('train average step error', train_error_step / len(dataset_train) , epoch)
        writer.add_scalar('train average rot_x acc', sum(train_rot_x_acc) / len(train_rot_x_acc) , epoch)
        writer.add_scalar('train average rot_y acc', sum(train_rot_y_acc) / len(train_rot_y_acc) , epoch)
        writer.add_scalar('train average rot_z acc', sum(train_rot_z_acc) / len(train_rot_z_acc) , epoch)
        
        writer.add_scalar('valid average rot error', valid_error_rot / len(dataset_valid) , epoch)
        writer.add_scalar('valid average xyz error', valid_error_xyz / len(dataset_valid) , epoch)
        writer.add_scalar('valid train average rot_x acc', sum(valid_rot_x_acc) / len(valid_rot_x_acc) , epoch)
        writer.add_scalar('valid train average rot_y acc', sum(valid_rot_y_acc) / len(valid_rot_y_acc) , epoch)
        writer.add_scalar('valid train average rot_z acc', sum(valid_rot_z_acc) / len(valid_rot_z_acc) , epoch)
    writer.close()

if __name__ == '__main__':
    
    checkpoint_dir = os.path.join(os.path.dirname(__file__), checkpoint_dir_name)
    #net_path = 'ckpnt_xyzrot_0725/checkpoint-100.pth'
    
    start_time = time.time()
    train(checkpoint_dir=checkpoint_dir)
    end_time = time.time()
    print('training time:' + str(end_time-start_time))

    # The visualization code
    #tmp_dir = 'tmp'
    #if not os.path.exists(tmp_dir):
    #   os.mkdir(tmp_dir)
    #visualize(net_path, tmp_dir)
