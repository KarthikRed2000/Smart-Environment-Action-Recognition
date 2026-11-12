"""ST-GCN (Yan et al., AAAI'18) plus Bi-LSTM and 1D-CNN baselines. Input: (N, C=3, T, V=25)."""
import numpy as np
import torch
import torch.nn as nn

# NTU / Kinect-v2 bones (1-indexed, as in the ST-GCN code)
_EDGES = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
          (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
          (19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]


def ntu_graph(V=25):
    """Spatial partitioning: self / inward (toward the center) / outward."""
    I = np.eye(V)
    inward = np.zeros((V, V))
    for i, j in _EDGES:
        inward[j - 1, i - 1] = 1          # child i -> parent j
    outward = inward.T

    def norm(A):
        d = A.sum(0)
        d[d == 0] = 1
        return A / d
    return torch.tensor(np.stack([I, norm(inward), norm(outward)]), dtype=torch.float32)


class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, residual=True, kt=9, dropout=0.0):
        super().__init__()
        self.K = A.shape[0]
        self.register_buffer("A", A)
        self.importance = nn.Parameter(torch.ones_like(A))   # learnable edge importance
        self.gcn = nn.Conv2d(cin, cout * self.K, 1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, (kt, 1), (stride, 1), ((kt - 1) // 2, 0)),
            nn.BatchNorm2d(cout), nn.Dropout(dropout))
        if not residual:
            self.res = lambda x: 0
        elif cin == cout and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, _, T, V = x.shape
        y = self.gcn(x).view(N, self.K, -1, T, V)
        y = torch.einsum("nkctv,kvw->nctw", y, self.A * self.importance)
        return self.relu(self.tcn(y) + self.res(x))


class STGCN(nn.Module):
    def __init__(self, num_class, in_c=3, V=25, dropout=0.1):
        super().__init__()
        A = ntu_graph(V)
        self.data_bn = nn.BatchNorm1d(in_c * V)
        cfg = [(in_c, 64, 1, False), (64, 64, 1, True), (64, 64, 1, True), (64, 64, 1, True),
               (64, 128, 2, True), (128, 128, 1, True), (128, 128, 1, True),
               (128, 256, 2, True), (256, 256, 1, True), (256, 256, 1, True)]
        self.blocks = nn.ModuleList(STGCNBlock(a, b, A, s, r, dropout=dropout) for a, b, s, r in cfg)
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        N, C, T, V = x.shape
        x = self.data_bn(x.permute(0, 3, 1, 2).reshape(N, V * C, T))
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        for b in self.blocks:
            x = b(x)
        return self.fc(x.mean((2, 3)))


class BiLSTM(nn.Module):
    def __init__(self, num_class, in_c=3, V=25, hidden=256, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_c * V, hidden, layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(2 * hidden, num_class)

    def forward(self, x):
        N, C, T, V = x.shape
        h, _ = self.lstm(x.permute(0, 2, 3, 1).reshape(N, T, V * C))
        return self.fc(h.mean(1))


class CNN1D(nn.Module):
    def __init__(self, num_class, in_c=3, V=25):
        super().__init__()
        def blk(a, b, s=1):
            return nn.Sequential(nn.Conv1d(a, b, 5, s, 2), nn.BatchNorm1d(b), nn.ReLU(inplace=True))
        self.net = nn.Sequential(blk(in_c * V, 128), blk(128, 128), blk(128, 256, 2),
                                 blk(256, 256), blk(256, 256, 2), nn.Dropout(0.2))
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        N, C, T, V = x.shape
        h = self.net(x.permute(0, 3, 1, 2).reshape(N, V * C, T))
        return self.fc(h.mean(-1))


MODELS = {"stgcn": STGCN, "bilstm": BiLSTM, "cnn1d": CNN1D}
