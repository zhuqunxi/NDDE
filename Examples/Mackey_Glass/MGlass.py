import torchdiffeq as ode
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cubic_spline import CSpline
import math
import warnings

warnings.filterwarnings('ignore')


class ODEFunc(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0):
        super(ODEFunc, self).__init__()
        self.dropout = dropout
        # self.tm = 3.18
        self.tm = 5

        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),
                                         nn.Linear(hidden_size, hidden_size, bias=True), nn.Tanh())

        self.dropout_layer = nn.Dropout(dropout)

        self.output_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=True), nn.Tanh(),
                                          nn.Linear(hidden_size, output_size, bias=True))

        self.fit_data = None

    def forward(self, t, x):
        x_tau = self.fit_data.fit(t - self.tm)
        x = torch.cat((x, x_tau), dim=-1)
        x = self.input_layer(x)
        x = self.dropout_layer(x)
        x = self.output_layer(x)
        return x


def read_data(filename):
    Xs = pd.read_csv("./dataset/data_" + filename + "/trajectory.csv").values[:, 1:]
    t = pd.read_csv("./dataset/data_" + filename + "/time_point.csv").values[:, 1]
    return (Xs, t)


def train(filename, input_size, hidden_size, output_size, dropout):
    odefunc = ODEFunc(input_size, hidden_size, output_size, dropout)
    odefunc.fit_data = CSpline(T, X[:, 0])

    batch_num = 20
    batch_size = 32
    params = odefunc.parameters()
    optimizer = optim.Adam(params, lr=0.01, weight_decay=1e-5)

    for epoch in range(epoch_num):
        loss = 0
        for batch in range(batch_num):
            optimizer.zero_grad()
            loss_batch = torch.tensor(0.0)
            for ib in range(batch_size):
                nt = np.random.randint(taum, tt - tle)
                true_y0 = X[nt, :].reshape(1, 1)
                pred_y = ode.odeint(odefunc, true_y0, T[nt:nt + tle],
                                    rtol=rtol, atol=atol, method=method)
                loss_batch = loss_batch + torch.sum((pred_y[:, 0] - X[nt:nt + tle]) ** 2)
            loss_batch = loss_batch / batch_size
            loss_batch.backward()
            optimizer.step()
            loss = loss + loss_batch.detach().numpy()
        print("epoch:", epoch, ", loss:", loss)
        if epoch >= 0:
            torch.save(odefunc, './model/node_' + filename + samp + str(tle) + '_' + str(epoch) + '.pkl')
            draw(tt, epoch)

    return (odefunc)


def draw(tt, epoch):
    odefunc = torch.load('./model/node_' + filename + samp + str(tle) + '_' + str(epoch) + '.pkl')
    nt = 500  # nt>taum
    le = 8000  # Value range [0, len(t)-nt]

    pred_y = np.zeros((le, V))
    # Piecewise prediction
    Pn = math.ceil(le / taum)
    tn = t[nt - taum:nt + 1]
    xn = torch.tensor(x[nt - taum:nt + 1, 0]).double()
    for pn in range(Pn):
        odefunc.fit_data = CSpline(tn, xn)
        true_y0 = xn[-1].reshape(1, 1)
        ttmin = min(nt + (pn + 1) * taum + 1, nt + le)
        tl = t[nt + pn * taum:ttmin]
        pred = ode.odeint(odefunc, true_y0, tl,
                          rtol=rtol, atol=atol, method=method)
        pred_y[pn * taum:ttmin - nt] = pred[:, 0].detach().numpy()
        tn = t[nt - taum + (pn + 1) * taum:nt + 1 + (pn + 1) * taum]
        xn = torch.tensor(pred_y[pn * taum:ttmin - nt, 0])

    Y = np.zeros((le - taum, 2))
    Y[:, 0] = pred_y[taum:, 0]
    Y[:, 1] = pred_y[:-taum, 0]
    Xnum = X.numpy()
    fig = plt.figure(figsize=(30, 12))
    gs = GridSpec(V, 2, figure=fig)
    font1 = {'weight': 'normal', 'size': 30}
    for i in range(V):
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(mask, Xnum[:, i], 'k-', linewidth=3)
        ax1.plot(np.arange(nt, nt + le), pred_y[:, i], 'r-', linewidth=2)
    ax2 = fig.add_subplot(gs[0:2, 1])
    ax2.plot(Xnum[nt + taum:, 0], Xnum[nt:-taum, 0], 'ko', markersize=5, linewidth=3, label='Ground Truth')
    if le + nt <= tt:
        plt.plot(Y[:, 0], Y[:, 1], 'r-', markersize=5, linewidth=3, label='Interpolation')
    else:
        plt.plot(Y[:tt - nt, 0], Y[:tt - nt, 1], 'r-', markersize=5, linewidth=3, label='Interpolation')
        plt.plot(Y[tt - nt:, 0], Y[tt - nt:, 1], 'b-', markersize=10, linewidth=3, label='Extrapolation')
    plt.legend(prop=font1)
    plt.savefig("./figures/MGlass_" + str(epoch) + samp + ".jpg")
    plt.close()


np.random.seed(0)

V = 1
# taum = 106
taum = 25
filename = 'MGlass'
samp = '_ir_'

x, t = read_data(filename)
t = torch.tensor(t).float()
X = [];
T = []
if samp == '_re_':
    X = x;
    T = t
    mask = np.arange(len(t))
elif samp == '_ir_':
    mask = []
    for i in range(len(t)):
        if np.random.rand() < 0.80:
            mask.append(i)
            X.append(x[i]);
            T.append(t[i])
X = torch.tensor(np.array(X)).double()
T = torch.tensor(np.array(T)).double()
tt = int(X.shape[0] * 0.95)

tle = 10
epoch_num = 128
rtol = 1e-5;
atol = 1e-7;
method = 'dopri5'
input_size = 2 * V;
output_size = V;
dropout = 0.0
hidden_size = 30
node = train(filename, input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=dropout)
print('done!')