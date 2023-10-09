import torch.nn as nn
import torch.nn.functional as F
from model.pointnet2.pointnet2_utils import PointNetSetAbstraction


class PointNet2(nn.Module):
    def __init__(self):
        super(PointNet2, self).__init__()

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=21, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        return l3_xyz, l3_points

if __name__ == '__main__':
    from torch.autograd import Variable
    import torch

    sim_data = Variable(torch.rand(2,3,21))
    model = PointNet2()
    x, y = model(sim_data)
