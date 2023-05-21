import math
import os
import numpy as np
import random
import torch
from torch.optim import Adam, Adagrad, SGD
from torch.distributions import Normal
from torch.distributions.gamma import Gamma
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import lending_loader as ld
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='o_langevin')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--thres', type=float, default=0.0)
parser.add_argument('--alpha', type=float, default=80)#o_langevin:alpha=80,lr=1e-4; o_svgd: a=100,lr=1e-3
parser.add_argument('--lr', type=float, default=1e-4)
## o_langevin
parser.add_argument('--useHessian', type=bool, default=True)
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
IDX_TRAIN = None
IDX_TEST = None
method = args.method
THRES = args.thres
NUM_PARTICLES = 10 
RUN = args.seed
device = torch.device('cuda')
if method == 'o_svgd':
    from svgd_constraint import get_gradient
    LR = args.lr
    EPOCH = 2000
elif method == 'o_svgd_fast': # ignore the second-order term for speedup
    from svgd_constraint import get_gradient_fast
    LR = args.lr
    EPOCH = 2000
elif method == 'svgd':
    from svgd_constraint import get_gradient_standard
    LR = args.lr
    EPOCH = 2000
elif method == 'o_langevin':
    from langevin_constraint import get_gradient
    LR = args.lr
    EPOCH = 2000
elif method == 'langevin':
    from langevin_constraint import get_gradient_standard
    LR = args.lr
    EPOCH = 2000
else:
    print('Not Available')
    assert False
method = 'loan_'+ method
method += '_%s'%args.useHessian + '_%d'%NUM_PARTICLES +'_%.3f'%args.alpha + '_%s'%LR + '_%d'%args.seed
log_dir = 'logs/%s/'%method
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
print(method)

idx = [i for i in range(28)]

class BayesianNN:
    def __init__(self, X_train, y_train, batch_size, num_particles, hidden_dim):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_particles = num_particles
        self.n_features = X_train.shape[1] 
        self.hidden_dim = hidden_dim

    def forward(self, inputs, theta, constraint = False):
        # Unpack theta
        w1 = theta[:, 0:self.n_features * self.hidden_dim].reshape(-1, self.n_features, self.hidden_dim)
        b1 = theta[:, self.n_features * self.hidden_dim:(self.n_features + 1) * self.hidden_dim].unsqueeze(1)

        # num_particles times of forward
        inputs = inputs.unsqueeze(0).repeat(theta.size(0), 1, 1)
        out_logit = torch.bmm(inputs, w1) + b1
        out = out_logit.squeeze()
        if constraint:
            return out, out_logit.squeeze()
        else:
            return out

    def get_log_prob_and_constraint(self, theta):
        model_w = theta[:, :]
        w_prior = Normal(0., 1.)

        random_idx = random.sample([i for i in range(self.X_train.shape[0])], self.batch_size)
        X_batch = self.X_train[random_idx]
        y_batch = self.y_train[random_idx]

        outputs, out_logit = self.forward(X_batch[:, idx], theta, constraint=True)  # [num_particles, batch_size]
        y_batch_repeat = y_batch.unsqueeze(0).repeat(self.num_particles, 1)
        log_p_data = F.binary_cross_entropy_with_logits(outputs, y_batch_repeat, reduction='none') 
        log_p_data = (-1.)*log_p_data.sum(dim=1)

        log_p0 = w_prior.log_prob(model_w.t()).sum(dim=0) 
        log_p = log_p0 + log_p_data * (self.X_train.shape[0] / self.batch_size) 

        global IDX_TRAIN, IDX_TEST
        small_samples = self.X_train[IDX_TRAIN] 
        outputs = self.forward(small_samples, theta)
        constrain = F.binary_cross_entropy_with_logits(outputs, torch.ones_like(outputs),reduction='none') - THRES 

        return log_p, constrain.mean(dim=1)

    def get_constraint(self, theta):
        global IDX_TRAIN, IDX_TEST
        small_samples = self.X_train[IDX_TRAIN] 
        outputs = self.forward(small_samples, theta)
        constrain = F.binary_cross_entropy_with_logits(outputs, torch.ones_like(outputs)) - THRES 

        return constrain

def test(model, theta, X_test, y_test):
    global IDX_TEST
    with torch.no_grad():
        prob = model.forward(X_test[:, idx], theta)
        y_pred = torch.sigmoid(prob).mean(dim=0)  # Average among outputs from different network parameters(particles)
        y_pred = y_pred.cpu().numpy()
        CV = y_pred[IDX_TEST]
        print(CV)
        CV[CV>=0.5] = 1.
        CV[CV<0.5] = 0.
        CV = 1. - CV.sum()/CV.shape[0]

        y_pred[y_pred>= 0.5] = 1
        y_pred[y_pred<0.5] = 0
        acc_bnn = np.sum(y_pred==y_test.cpu().numpy())/float(y_test.shape[0])
        cv_bnn = CV
        print('acc: ', acc_bnn, 'test violation:', CV)
       
        acc_cllt = []
        cv_cllt = []
        for i in range(prob.shape[0]):
            y_pred = torch.sigmoid(prob[i, :])  # Average among outputs from different network parameters(particles)
            y_pred = y_pred.cpu().numpy()
            CV = y_pred[IDX_TEST] > 0.5
            CV = 1. - CV.sum()/np.sum(IDX_TEST)
            y_pred[y_pred>= 0.5] = 1
            y_pred[y_pred<0.5] = 0
            acc_cllt.append(np.sum(y_pred==y_test.cpu().numpy())/float(y_test.shape[0]))
            cv_cllt.append(CV)
        print('mean CV:', np.mean(np.array(cv_cllt)))

        return acc_cllt, cv_cllt, acc_bnn , cv_bnn


def main():
    global IDX_TRAIN, IDX_TEST
    X_train, y_train, X_test, y_test, start_index, cat_length = ld.load_data(get_categorical_info=True)
    X_train=X_train[:20000]
    y_train=y_train[:20000]
    n = X_train.shape[0]
    n = int(0.99*n)
    X_val = X_train[n:, :]
    y_val = y_train[n:]
    X_train = X_train[:n, :]
    y_train = y_train[:n]

    feature_num = X_train.shape[1]
    X_train = torch.tensor(X_train).float().to(device)
    X_test = torch.tensor(X_test).float().to(device)
    X_val = torch.tensor(X_val).float().to(device)
    y_train = torch.tensor(y_train).float().to(device)
    y_test = torch.tensor(y_test).float().to(device)
    y_val = torch.tensor(y_val).float().to(device)

    a = X_train[:, 0] == 1.
    b = X_train[:, 2] == 1.
    IDX_TRAIN = (a*b).cpu().numpy()
    c = X_test[:, 0] == 1.
    d = X_test[:, 2] == 1.
    IDX_TEST = (c*d).cpu().numpy()
    print('STAT:', IDX_TRAIN.sum(), IDX_TEST.sum(), y_train[IDX_TRAIN].sum(), y_test[IDX_TEST].sum())

    X_train_mean, X_train_std = torch.mean(X_train[:, idx], dim=0), torch.std(X_train[:, idx], dim=0)
    X_train[:, idx] = (X_train [:, idx]- X_train_mean) / X_train_std
    X_test[:, idx] = (X_test[:, idx] - X_train_mean) / X_train_std

    num_particles, batch_size, hidden_dim = NUM_PARTICLES, 1000, 1

    model = BayesianNN(X_train, y_train, batch_size, num_particles, hidden_dim)

    # Random initialization (based on expectation of gamma distribution)
    theta = torch.cat([torch.zeros([num_particles, (X_train.shape[1] + 2) * hidden_dim + 1], device=device).normal_(0, math.sqrt(0.05))])
    
    cls_loss_cllt = []
    c_loss_cllt = []
    lmbda_cllt = []
    acc_bnn_cllt = []
    cv_bnn_cllt = []
    lr = LR
    optim_c = 1e6

    if 'langevin' in method:
        optimizer = SGD([theta], lr=1., weight_decay=0.0)
    else:
        optimizer = SGD([theta], lr=lr, weight_decay=0.0)

    
    for epoch in range(EPOCH):
        b = 1.0
        gamma = -0.55
        lr = LR * ((b + epoch)**(gamma))
        
        if 'svgd' in method:
            for g in optimizer.param_groups:
                g['lr'] = lr
       
        if args.method == 'o_svgd':
            optimizer.zero_grad()
            theta.grad, cls_loss, c_loss, cls_loss_ind = get_gradient(model, theta,alpha=args.alpha)
            optimizer.step()
        elif args.method == 'o_svgd_fast':
            optimizer.zero_grad()
            theta.grad, cls_loss, c_loss, cls_loss_ind = get_gradient_fast(model, theta,alpha=args.alpha)
            optimizer.step()
        elif args.method == 'svgd':
            optimizer.zero_grad()
            theta.grad, cls_loss, c_loss, cls_loss_ind = get_gradient_standard(model, theta,alpha=args.alpha)
            optimizer.step()
        elif args.method == 'o_langevin':
            theta.grad, cls_loss, c_loss, cls_loss_ind = get_gradient(model, theta, lr,alpha=args.alpha,useHessian=args.useHessian)
            optimizer.step()
        elif args.method == 'langevin':
            theta.grad, cls_loss, c_loss, cls_loss_ind = get_gradient_standard(model, theta, lr,alpha=args.alpha,useHessian=args.useHessian)
            optimizer.step()
        
        if (c_loss.item()<optim_c) and (epoch>(EPOCH - 400)):
            optim_c = c_loss.item()
            optim_theta = theta.detach().clone()


        cls_loss_cllt.append(cls_loss.item())
        c_loss_cllt.append(c_loss.item())
        if epoch % 100 == 0:
            print('cls loss:', cls_loss.item(), 'c loss:', c_loss.item())
            acc_cllt, cv_cllt, acc_bnn, cv_bnn = test(model, theta, X_test, y_test)
            acc_bnn_cllt.append(acc_bnn)
            cv_bnn_cllt.append(cv_bnn)


    acc_cllt, cv_cllt, acc_bnn, cv_bnn = test(model, theta, X_test, y_test)
    acc_bnn_cllt.append(acc_bnn)
    cv_bnn_cllt.append(cv_bnn)
    
    print('Optimal Theta:')
    acc_cllt, cv_cllt, acc_bnn, cv_bnn = test(model, optim_theta, X_test, y_test)
    np.save('%s/acc_ct.npy'%(log_dir), np.array(acc_cllt))
    np.save('%s/cv_ct.npy'%(log_dir), np.array(cv_cllt))
    np.save('%s/acc_bnn_ct.npy'%(log_dir), np.array(acc_bnn_cllt))
    np.save('%s/cv_bnn_ct.npy'%(log_dir), np.array(cv_bnn_cllt))
    np.save('%s/cls_ct.npy'%(log_dir), np.array(cls_loss_cllt))
    np.save('%s/c_loss_ct.npy'%(log_dir), np.array(c_loss_cllt))
    print('VAR:', torch.sqrt((cls_loss_ind/1000.).var()), (cls_loss_ind/batch_size).mean())

if __name__ == '__main__':
    main()
