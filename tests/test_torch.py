from GitMarco.torch.losses import ChamferLoss
import numpy as np
import torch


def test_chamfer_loss():
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    x = y = np.arange(100, )
    z1 = np.random.rand(100, )
    z2 = np.random.rand(100, )
    true = torch.unsqueeze(torch.Tensor(np.vstack((x, y, z1))).to(device), dim=0)
    pred = torch.unsqueeze(torch.Tensor(np.vstack((x, y, z2))).to(device), dim=0)
    criterion = ChamferLoss()
    criterion(true, pred)


