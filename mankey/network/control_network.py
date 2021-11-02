import torch
import torch.nn as nn
import sys
sys.path.append('/tmp2/r09944001/robot-peg-in-hole-task')
from mankey.network.utils import compute_rotation_matrix_from_ortho6d

class ControlNetwork(nn.Module):
    def __init__(self, in_channel):
        super(ControlNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.mlp(x)
        out_r_6d = x[:, 0:6]
        out_r = compute_rotation_matrix_from_ortho6d(out_r_6d) # batch*3*3
        out_t = x[:, 6:9].view(-1,3) # batch*3*1
        out_step_size = x[:, 9].view(-1,1)# batch*1
        return out_r, out_t, out_step_size
