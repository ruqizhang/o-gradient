import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import torch
import seaborn as sns
sns.set(color_codes=True)
sns.set_style("white")
import matplotlib.animation as animation
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from scipy.stats import wasserstein_distance
import dcor
import samplers
import time

def plot_2d_contour(f, xlim, ylim, gridsize=100): # f is an 2D function.     
    x = np.linspace(xlim[0], xlim[1], gridsize)
    y = np.linspace(ylim[0], ylim[1], gridsize)
    X, Y = np.meshgrid(x, y)
    XY = np.vstack([X.ravel(), Y.ravel()]).T
    Zf = f(XY).reshape((gridsize, gridsize))
    c = plt.contour(X, Y, Zf) 
    return 


def plot_2d_zero(f, xlim, ylim, gridsize=100): # f is an 2D function.     
    x = np.linspace(xlim[0], xlim[1], gridsize)
    y = np.linspace(ylim[0], ylim[1], gridsize)
    X, Y = np.meshgrid(x, y)
    XY = np.vstack([X.ravel(), Y.ravel()]).T
    Zf = f(XY).reshape((gridsize, gridsize))
    c = plt.contour(X, Y, Zf, 0, colors='red') 
    return c

def compute_w2(x):
  gt = 2.5**.5*np.random.randn(1000)+1
  w2 = wasserstein_distance(gt,x[:,0])
  # print('w2',w2)
  return w2

def compute_ed(x,gt):
    ed = dcor.energy_distance(gt,x)
    return ed

def main(args,gt):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dim = 2
    NUM_PARTICLES = 50
    if args.init=='on':
        x00=8
        x01=-2
        x0 = torch.zeros(NUM_PARTICLES,dim)
        x0[:,1] = x01 + torch.randn(NUM_PARTICLES)*0.1
        x0[:,0] = -x0[:,1]**3
    else:
        x00=-10
        x01=1
        x0 = torch.zeros(NUM_PARTICLES,dim)+x01
        x0[:,0] = x00
        x0 += torch.randn(NUM_PARTICLES,dim)*0.1

    if args.sampler == 'o_langevin':
        sampler = samplers.O_Langevin(logp, g, stepsize=args.lr, alpha = args.alpha, beta=1, useHessian=args.useHessian, M = 1000)
    elif args.sampler == 'o_svgd':
        sampler = samplers.O_SVGD(logp, g, stepsize=args.lr,alpha = args.alpha, M = 1000)
    
    xs = []
    w2 = []
    gx=[]
    xs.append(x0.detach().unsqueeze(0))
    x=x0
    dim = x.size
    total_time = 0
    for i in range(1,args.burnin+args.maxiter+1):
        start = time.time()
        x = sampler.step(x)
        total_time += time.time()-start        
        if i>args.burnin:
            xs.append(x.detach().unsqueeze(0))  
            if i%10==0:
                w2.append(compute_ed(x.detach().numpy(),gt)) 
                gx.append(np.absolute(g(x.detach().numpy())).mean())               
    xs = torch.cat(xs,dim=0)
    return xs.numpy(), w2, total_time,gx

def logp(x): 
  z0 = x[:,0]+x[:,1]**3
  z1 = x[:,1]
  return -((z0)**2/2 + (z1)**2/2)

def g(x): 
  return x[:,0]+x[:,1]**3
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', type=str, default='on') # whether the initialization is on the manifold 
    parser.add_argument('--sampler', type=str, default='o_langevin')
    parser.add_argument('--maxiter', type=int, default=5000)
    parser.add_argument('--burnin', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--alpha', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2) # o_svgd:0.5
    ## o_langevin
    parser.add_argument('--useHessian', type=bool, default=True)

    args = parser.parse_args()
    num=50
    gt = np.zeros((num,2))
    z = np.random.randn(num)
    gt[:,0] = -z**3
    gt[:,1] = z

    xs,w2,total_time,gx = main(args,gt)
    print('runtime',total_time)
    bound = 40
    boundy = 4

    plt.clf()
    np.save("figs/%s_samples_%s_%s_burnin%d_s%d_a%d_lr%s.npy"%(args.init,args.sampler,args.useHessian,args.burnin,args.seed,args.alpha,args.lr),xs)
    plt.plot(w2)
    plt.xlabel('Iters ',fontsize=17)
    plt.ylabel('Energy Dist',fontsize=17)
    plt.savefig('figs/%s_ed_%s_%s_burnin%d_s%d_a%d_lr%s.pdf'%(args.init,args.sampler,args.useHessian,args.burnin,args.seed,args.alpha,args.lr))
    np.save("figs/%s_ed_%s_%s_burnin%d_s%d_a%d_lr%s.npy"%(args.init,args.sampler,args.useHessian,args.burnin,args.seed,args.alpha,args.lr),w2)

    plt.clf()
    plot_2d_contour(logp, xlim=[-bound,bound], ylim=[-boundy,boundy], gridsize=100)
    plot_2d_zero(g, xlim=[-bound,bound], ylim=[-boundy,boundy], gridsize=100)
    plt.plot(xs[-1,:,0], xs[-1,:,1], '.', alpha=0.8, markersize=5)
    plt.savefig('figs/%s_path_%s_%s_burnin%d_s%d_a%d_lr%s.pdf'%(args.init,args.sampler,args.useHessian,args.burnin,args.seed,args.alpha,args.lr))

    plt.clf()
    plt.plot(gx)
    plt.xlabel('Iters ',fontsize=17)
    plt.ylabel('Constraint',fontsize=17)
    np.save('figs/%s_g_%s_%s_burnin%d_s%d_a%d_lr%s.npy'%(args.init,args.sampler,args.useHessian,args.burnin,args.seed,args.alpha,args.lr),gx)
    plt.savefig('figs/%s_g_%s_%s_burnin%d_s%d_a%d_lr%s.pdf'%(args.init,args.sampler,args.useHessian,args.burnin,args.seed,args.alpha,args.lr))