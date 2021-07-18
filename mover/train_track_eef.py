import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time
import random
import logging
import torch
import numpy as np
from tqdm import tqdm
from models.model_track_eef import Model, init_from_modelzoo
from losses.loss import RMSELoss
from dataset.dataset_track import RobotDataset
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter


seed = 73
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

writer = SummaryWriter()
logging.basicConfig(level=logging.INFO)

# Some global parameter
learning_rate = 2e-4
n_epoch = 300
batch_size = 1
early_stop = 100
w_r = 0.0
w_t = 1.0
w_s = 0.0
w_next_t = 1.0
data_folder = '/tmp2/r09944001/data/pdc/logs_proto/xyz_track_curve_insertion_2021-06-20'
checkpoint_dir = os.path.join(os.path.dirname(__file__), 'ckpnt')
checkpoints_best_file_path = os.path.join(checkpoint_dir, 'checkpoint-best.pth')

resnet_num_layers = 18
image_channels = 4

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # The checkpoint
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    dataset = RobotDataset(data_folder)
    # Random split
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size
    train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    model = Model(resnet_layers=resnet_num_layers, in_channel=image_channels)
    if os.path.isfile(checkpoints_best_file_path):
        model.load_state_dict(torch.load(checkpoints_best_file_path))
    else:
        init_from_modelzoo(model=model, resnet_num_layers=resnet_num_layers, image_channels=image_channels)
        print('init from modelzoo !')
    model.to(device)

    # root mean square error loss
    criterion_rmse_t = RMSELoss(reduction='none')
    criterion_rmse_r = RMSELoss()
    criterion_cos = torch.nn.CosineSimilarity(dim=1)
    criterion_bce = torch.nn.BCELoss(reduction='none')
    # The optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 90], gamma=0.1)

    min_rmse = 1000.
    for epoch in range(1, n_epoch+1):
        if epoch % 20 == 0 and epoch > 0:
            file_name = 'checkpoint-{:d}.pth'.format(epoch)
            checkpoint_path = os.path.join(checkpoint_dir, file_name)
            logging.info('Save model at {:s}'.format(checkpoint_path))
            torch.save(model.state_dict(), checkpoint_path)

        model.train()
        # The learning rate step
        #scheduler.step()
        #for param_group in optimizer.param_groups:
        #    logging.info('The learning rate is {:f}'.format(param_group['lr']))

        train_loss = []
        train_loss_r = []
        train_loss_t = []
        train_loss_step = []
        # The training iteration over the dataset
        progress = tqdm(train_loader, desc='train epoch {:d}'.format(epoch), leave = False)
        for idx, (rgbd_track, gt_r_track, gt_t_track, gt_step_track, gt_pos_track, gt_next_t_track) in enumerate(progress):
            train_loss_ep = []
            train_loss_r_ep = []
            train_loss_t_ep = []
            train_loss_step_ep = []
            num_episode = len(rgbd_track)
            for episode in range(num_episode):
                rgbd = rgbd_track[episode]
                gt_r = gt_r_track[episode]
                gt_t = gt_t_track[episode]
                gt_step = gt_step_track[episode]
                gt_next_t = gt_next_t_track[episode]
                gt_pos = gt_pos_track[episode]

                out_r, out_t, step, out_next_t = model(rgbd.to(device),gt_pos.to(device))
                loss_r = criterion_rmse_r(out_r, gt_r.to(device))
                loss_step = criterion_bce(step, gt_step.to(device))
                loss_step = loss_step.mean()
                #loss_t = torch.mul(step, (1-criterion_cos(out_t, gt_t.to(device))).mean()) + loss_step
                #loss_t = (1-criterion_cos(out_t, gt_t.to(device))).mean() + loss_step
                #print(out_t.shape, out_next_t.shape)
                weight_x = 1/3
                weight_y = 1/3
                weight_z = 1/3
                print('out_t',out_t)
                print('gt_t',gt_t)
                loss_t = criterion_rmse_t(out_t, gt_t.to(device))
                print(loss_t)

                loss_t = loss_t[:,0] * weight_x +  loss_t[:,1] * weight_y +  loss_t[:,2] * weight_z 
                loss_t = loss_t.mean()

                loss_next_t = criterion_rmse_t(out_next_t, gt_next_t.to(device))
                loss_next_t = loss_next_t[:,0] * weight_x +  loss_next_t[:,1] * weight_y +  loss_next_t[:,2] * weight_z
                loss_next_t = loss_next_t.mean()
                loss = loss_t * w_t + loss_next_t * w_next_t

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress.set_postfix({'loss':loss.item(),
                                      'loss_r':loss_r.item(), 
                                      'loss_t':loss_t.item(), 
                                      'loss_step':loss_step.mean().item()})
                train_loss_ep.append(loss.item())
                train_loss_r_ep.append(loss_r.item())
                train_loss_t_ep.append(loss_t.item())
                train_loss_step_ep.append(loss_step.mean().item())
            train_loss_ep = sum(train_loss_ep) / len(train_loss_ep)
            train_loss_r_ep = sum(train_loss_r_ep) / len(train_loss_r_ep)
            train_loss_t_ep = sum(train_loss_t_ep) / len(train_loss_t_ep)
            train_loss_step_ep = sum(train_loss_step_ep) / len(train_loss_step_ep)
            train_loss.append(train_loss_ep)
            train_loss_r.append(train_loss_r_ep)
            train_loss_t.append(train_loss_t_ep)
            train_loss_step.append(train_loss_step_ep)
        train_loss = sum(train_loss) / len(train_loss)
        train_loss_r = sum(train_loss_r) / len(train_loss_r)
        train_loss_t = sum(train_loss_t) / len(train_loss_t)
        train_loss_step = sum(train_loss_step) / len(train_loss_step)

        logging.info('training loss: {:.4f}'.format(train_loss))
        #logging.info('training loss step: {:.4f}'.format(train_loss_step))
        #logging.info('training loss_r: {:.4f}'.format(train_loss_r))
        logging.info('training loss_t: {:.4f}'.format(train_loss_t))
        writer.add_scalar('training_loss', train_loss, epoch)
        writer.add_scalar('training_loss_r', train_loss_r, epoch)
        writer.add_scalar('training_loss_t', train_loss_t, epoch)
        writer.add_scalar('training_loss_step', train_loss_step, epoch)

        valid_loss = []
        valid_loss_r = []
        valid_loss_t = []
        valid_loss_step = []

        # validation
        model.eval()
        progress = tqdm(valid_loader, desc='validation', leave=False)
        with torch.no_grad():
            for idx, (rgbd_track, gt_r_track, gt_t_track, gt_step_track, gt_pos_track, gt_next_t_track) in enumerate(progress):
                valid_loss_ep = []
                valid_loss_r_ep = []
                valid_loss_t_ep = []
                valid_loss_step_ep = []
                num_episode = len(rgbd_track)
                for episode in range(num_episode):
                    rgbd = rgbd_track[episode]
                    gt_r = gt_r_track[episode]
                    gt_t = gt_t_track[episode]
                    gt_step = gt_step_track[episode]
                    gt_next_t = gt_next_t_track[episode]
                    gt_pos = gt_pos_track[episode]

                    out_r, out_t, step, out_next_t = model(rgbd.to(device),gt_pos.to(device))
                    loss_r = criterion_rmse_r(out_r, gt_r.to(device))
                    loss_step = criterion_bce(step, gt_step.to(device))
                    loss_step = loss_step.mean()
                    #loss_t = torch.mul(step, (1-criterion_cos(out_t, gt_t.to(device))).mean()) + loss_step
                    #loss_t = (1-criterion_cos(out_t, gt_t.to(device))).mean() + loss_step

                    weight_x = 1/3
                    weight_y = 1/3
                    weight_z = 1/3
                    loss_t = criterion_rmse_t(out_t, gt_t.to(device))
                    loss_t = loss_t[:,0] * weight_x +  loss_t[:,1] * weight_y +  loss_t[:,2] * weight_z 
                    loss_t = loss_t.mean()
                    loss_next_t = criterion_rmse_t(out_next_t, gt_next_t.to(device))
                    loss_next_t = loss_next_t[:,0] * weight_x +  loss_next_t[:,1] * weight_y +  loss_next_t[:,2] * weight_z
                    loss_next_t = loss_next_t.mean()
                    loss = loss_t * w_t + loss_next_t * w_next_t
                    
                    progress.set_postfix({'loss': loss.item(), 
                                          'loss_r': loss_r.item(), 
                                          'loss_t': loss_t.item(),
                                          'loss_step': loss_step.mean().item()})
                    valid_loss_ep.append(loss.item())
                    valid_loss_r_ep.append(loss_r.item())
                    valid_loss_t_ep.append(loss_t.item())
                    valid_loss_step_ep.append(loss_step.mean().item())
                valid_loss_ep = sum(valid_loss_ep) / len(valid_loss_ep)
                valid_loss_r_ep = sum(valid_loss_r_ep) / len(valid_loss_r_ep)
                valid_loss_t_ep = sum(valid_loss_t_ep) / len(valid_loss_t_ep)
                valid_loss_step_ep = sum(valid_loss_step_ep) / len(valid_loss_step_ep)
                valid_loss.append(valid_loss_ep)
                valid_loss_r.append(valid_loss_r_ep)
                valid_loss_t.append(valid_loss_t_ep)
                valid_loss_step.append(valid_loss_step_ep)
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_loss_r = sum(valid_loss_r) / len(valid_loss_r)
        valid_loss_t = sum(valid_loss_t) / len(valid_loss_t)
        valid_loss_step = sum(valid_loss_step) / len(valid_loss_step)
        
        logging.info('validation loss: {:.4f}'.format(valid_loss))
        #logging.info('validation loss step: {:.4f}'.format(valid_loss_step))
        #logging.info('validation loss_r: {:.4f}'.format(valid_loss_r))
        logging.info('validation loss_t: {:.4f}'.format(valid_loss_t))
        writer.add_scalar('validation_loss', valid_loss, epoch)
        writer.add_scalar('validation_loss_r', valid_loss_r, epoch)
        writer.add_scalar('validation_loss_t', valid_loss_t, epoch)
        writer.add_scalar('validation_loss_step', valid_loss_step, epoch)
        logging.info('end of epoch {:d}'.format(epoch))

        if valid_loss < min_rmse:
            min_rmse = valid_loss
            logging.info('Saving model (epoch = {:4d}, loss = {:.4f})'.format(epoch, min_rmse))
            torch.save(model.state_dict(), checkpoints_best_file_path)
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt > early_stop:
            logging.info('early stop at epoch {:d}'.format(epoch))
            break

    writer.close()

if __name__ == '__main__':
    start_time = time.time()
    train()
    end_time = time.time()
    print('Time elapsed: {:.02f} hr'.format((end_time - start_time) / 3600))
