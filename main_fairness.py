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
import adult_loader as ad
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='o_langevin')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--thres', type=float, default=0.0)
parser.add_argument('--alpha', type=float, default=100)#o_langevin:alpha=100,lr=2e-5; o_svgd: a=130,lr=1e-4
parser.add_argument('--lr', type=float, default=2e-5)
## o_langevin
parser.add_argument('--useHessian', type=bool, default=True)
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
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

method += '_g2'+'_%s'%args.useHessian +'_%d'%NUM_PARTICLES +'_%.3f'%THRES +'_%.3f'%args.alpha + '_%s'%LR + '_%d'%args.seed
log_dir = 'logs/%s'%method
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
print(method)

idx = [i for i in range(87)]
del idx[45]

class BayesianNN:
    def __init__(self, X_train, y_train, batch_size, num_particles, hidden_dim):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_particles = num_particles
        self.n_features = X_train.shape[1] - 1
        self.hidden_dim = hidden_dim

    def forward(self, inputs, theta, constraint = False):
        # Unpack theta
        w1 = theta[:, 0:self.n_features * self.hidden_dim].reshape(-1, self.n_features, self.hidden_dim)
        b1 = theta[:, self.n_features * self.hidden_dim:(self.n_features + 1) * self.hidden_dim].unsqueeze(1)
        w2 = theta[:, (self.n_features + 1) * self.hidden_dim:(self.n_features + 2) * self.hidden_dim].unsqueeze(2)
        b2 = theta[:, -3].reshape(-1, 1, 1)

        # num_particles times of forward
        inputs = inputs.unsqueeze(0).repeat(theta.size(0), 1, 1)
        inter = F.relu(torch.bmm(inputs, w1) + b1)
        out_logit = torch.bmm(inter, w2) + b2
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

        ### NOTE: compute fairness loss
        mean_sense   = X_batch[:, 45].mean()
        weight_sense = X_batch[:, 45] - mean_sense
        out_logit = torch.sigmoid(out_logit)
        out_logit = torch.log((out_logit/(1.-out_logit+1e-6))+1e-6)
        out_logit = torch.sigmoid(out_logit)
        constrain = ((weight_sense * out_logit).mean(dim=1))**2 - THRES
        return log_p, constrain

    def get_constraint(self, theta):
        model_w = theta[:, :]

        random_idx = random.sample([i for i in range(self.X_train.shape[0])], self.batch_size)
        X_batch = self.X_train[random_idx]
        y_batch = self.y_train[random_idx]

        outputs, out_logit = self.forward(X_batch[:, idx], theta, constraint=True)  # [num_particles, batch_size]
        ### NOTE: compute fairness loss
        mean_sense   = X_batch[:, 45].mean()
        weight_sense = X_batch[:, 45] - mean_sense
        out_logit = torch.sigmoid(out_logit)
        out_logit = torch.log((out_logit/(1.-out_logit+1e-6))+1e-6)
        out_logit = torch.sigmoid(out_logit)
        constrain = ((weight_sense * out_logit).mean())**2 - THRES
        
        return constrain

def test(model, theta, X_test, y_test):
    with torch.no_grad():
        prob = model.forward(X_test[:, idx], theta)
        y_pred = torch.sigmoid(prob).mean(dim=0)  # Average among outputs from different network parameters(particles)
        y_pred = y_pred.cpu().numpy()
        sum_positive = np.zeros(2).astype(float)
        count_group = np.zeros(2).astype(float) 
        for j in range(2):
            A = y_pred[X_test.cpu().numpy()[:,45]==j]
            count_group[j] = A.shape[0]  
            sum_positive[j] = np.sum(A >= 0.5)
        ratio = sum_positive/count_group
        CV = np.max(ratio) - np.min(ratio)

        y_pred[y_pred>= 0.5] = 1
        y_pred[y_pred<0.5] = 0
        acc_bnn = np.sum(y_pred==y_test.cpu().numpy())/float(y_test.shape[0])
        cv_bnn = CV
        print('acc: ', np.sum(y_pred==y_test.cpu().numpy())/float(y_test.shape[0]), 'fairness:', CV)
       
        acc_cllt = []
        cv_cllt = []
        for i in range(prob.shape[0]):
            y_pred = torch.sigmoid(prob[i, :])  # Average among outputs from different network parameters(particles)
            y_pred = y_pred.cpu().numpy()
            sum_positive = np.zeros(2).astype(float)
            count_group = np.zeros(2).astype(float) 
            for j in range(2):
                A = y_pred[X_test.cpu().numpy()[:,45]==j]
                count_group[j] = A.shape[0]  
                sum_positive[j] = np.sum(A >= 0.5)
            ratio = sum_positive/count_group
            CV = np.max(ratio) - np.min(ratio)

            y_pred[y_pred>= 0.5] = 1
            y_pred[y_pred<0.5] = 0
            acc_cllt.append(np.sum(y_pred==y_test.cpu().numpy())/float(y_test.shape[0]))
            cv_cllt.append(CV)
        print('mean CV:', np.mean(np.array(cv_cllt)))

        return acc_cllt, cv_cllt, acc_bnn , cv_bnn


def main():
    X_train, y_train, X_test, y_test, start_index, cat_length = ad.load_data(get_categorical_info=True)
    X_train=X_train[:20000]
    y_train=y_train[:20000]
    n = X_train.shape[0]
    n = int(0.99*n)
    X_val = X_train[n:, :]
    y_val = y_train[n:]
    X_train = X_train[:n, :]
    y_train = y_train[:n]
    X_train = np.delete(X_train, 46, axis=1)
    X_val = np.delete(X_val, 46, axis=1)
    X_test = np.delete(X_test, 46, axis=1)

    feature_num = X_train.shape[1]
    X_train = torch.tensor(X_train).float().to(device)
    X_test = torch.tensor(X_test).float().to(device)
    X_val = torch.tensor(X_val).float().to(device)
    y_train = torch.tensor(y_train).float().to(device)
    y_test = torch.tensor(y_test).float().to(device)
    y_val = torch.tensor(y_val).float().to(device)

    X_train_mean, X_train_std = torch.mean(X_train[:, idx], dim=0), torch.std(X_train[:, idx], dim=0)
    X_train[:, idx] = (X_train [:, idx]- X_train_mean) / X_train_std
    X_test[:, idx] = (X_test[:, idx] - X_train_mean) / X_train_std
    
    num_particles, batch_size, hidden_dim = NUM_PARTICLES, 19800, 50

    model = BayesianNN(X_train, y_train, batch_size, num_particles, hidden_dim)

    # Random initialization 
    theta = torch.cat([torch.zeros([num_particles, (X_train.shape[1] -1 + 2) * hidden_dim + 1], device=device).normal_(0, math.sqrt(0.01))]) 
    
    cls_loss_cllt = []
    lmbda_cllt = []
    acc_bnn_cllt = []
    cv_bnn_cllt = []
    c_loss_max = []
    lr = LR
    optim_c = 1e6

    if 'langevin' in method:
        optimizer = SGD([theta], lr=1.)
    else:
        optimizer = SGD([theta], lr=lr)
    

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
            gradient, cls_loss, c_loss, cls_loss_ind = get_gradient(model, theta, lr,alpha=args.alpha,useHessian=args.useHessian)
            theta.grad = gradient.detach().clone()
            optimizer.step()
        elif args.method == 'langevin':
            gradient, cls_loss, c_loss, cls_loss_ind = get_gradient_standard(model, theta, lr,alpha=args.alpha,useHessian=args.useHessian)
            theta.grad = gradient.detach().clone()
            optimizer.step()
        
        if (c_loss.item()<optim_c) and (epoch>(EPOCH - 400)):
            optim_c = c_loss.item()
            optim_theta = theta.detach().clone()


        cls_loss_cllt.append(cls_loss.item())
        if epoch % 100 == 0:
            print('cls loss:', cls_loss.item(), 'c loss:', c_loss.item())
            acc_cllt, cv_cllt, acc_bnn, cv_bnn = test(model, theta, X_test, y_test)
            acc_bnn_cllt.append(acc_bnn)
            cv_bnn_cllt.append(cv_bnn)

            print('VAR:', torch.sqrt((cls_loss_ind/batch_size).var()), (cls_loss_ind/batch_size).mean())

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


if __name__ == '__main__':
    main()