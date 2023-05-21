import math
import torch
import numpy as np


def get_gradient(model, inputs, step_size, alpha=100,useHessian=True):
    n = inputs.size(0)
    dim = inputs.size(1)
    inputs = inputs.detach().requires_grad_(True)

    log_prob, constraint = model.get_log_prob_and_constraint(inputs)
    log_prob_grad = torch.autograd.grad(log_prob.sum(), inputs, allow_unused=True, retain_graph=True)[0]
    constraint_grad = torch.autograd.grad(constraint.sum(), inputs, allow_unused=True, create_graph=True)[0]

    v = step_size*log_prob_grad + np.sqrt(2*step_size) * torch.randn(log_prob_grad.shape, device = log_prob_grad.device)
    g_perp, g_para = project_g(v,  constraint_grad)
    phi = (alpha*constraint).unsqueeze(1).repeat(1,dim)
    Dgx2 = torch.sum(constraint_grad**2,dim=1,keepdim=True).repeat(1,dim)    
    if useHessian: 
        DxD = torch.zeros_like(inputs)
        # calculating the Hessian term. 
        for j in range(inputs.shape[0]):
            term1=torch.sum(constraint_grad[j,:]**2)
            tDgx = constraint_grad[j,:]/term1
            Hgx = torch.autograd.functional.hessian(model.get_constraint, inputs[[j],:]).squeeze()
            term3 = 2*torch.sum(tDgx @ Hgx * tDgx) * constraint_grad[j,:]
            DxD[j,:] = tDgx @ Hgx  +  tDgx * torch.trace(Hgx) - term3
        dx = g_perp - step_size*phi*constraint_grad/Dgx2 - step_size*DxD
    else:
        dx =  g_perp - step_size*phi*constraint_grad/Dgx2
    return -dx, log_prob.sum(), constraint.mean(), log_prob

def project_g(v, dg):
    proj = torch.sum(v*dg,dim=1)/torch.sum(dg**2,dim=1)
    g_para =proj.unsqueeze(1).repeat(1,v.size(1))*dg
    g_perp = v - g_para
    return g_perp, g_para