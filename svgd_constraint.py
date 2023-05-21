import math
import torch

def get_gradient(model, inputs,alpha=100):
    n = inputs.size(0)
    inputs = inputs.detach().requires_grad_(True)
    log_prob, constraint = model.get_log_prob_and_constraint(inputs)
    log_prob_grad = torch.autograd.grad(log_prob.sum(), inputs, allow_unused=True, retain_graph=True)[0]    


    c_list = []
    for i in range(n):
        constraint_grad = torch.autograd.grad(constraint[i].sum(), inputs, allow_unused=True, create_graph=True)[0]
        constraint_grad = constraint_grad[i].unsqueeze(1)
        
        g_norm_sqr = constraint_grad.norm().pow(2)
        D = torch.eye(constraint_grad.shape[0],device=inputs.device) - constraint_grad@constraint_grad.t() / g_norm_sqr
        c_list.append((D, constraint[i], constraint_grad))
    
    rbf_kernel_matrix = kernel_rbf(inputs)
    
    svgd_gradient = torch.zeros_like(inputs, device=inputs.device)
    
    for i in range(n):
        svgd_gradient[i, :] = get_single_particle_gradient_with_rbf_and_c(\
                            i, inputs, log_prob_grad, rbf_kernel_matrix, c_list, alpha).detach().clone()
    
    gradient = svgd_gradient / n

    
    return gradient, log_prob.mean(), constraint.mean(), log_prob

def get_single_particle_gradient_with_rbf_and_c(idx, inputs, log_prob_grad, rbf_kernel_matrix, c_list, alpha):
    n = inputs.size(0)
    d = inputs.shape[1]
    grad = None
    for j in range(n):
        K_rbf = rbf_kernel_matrix[idx, j] * torch.eye(d, device=inputs.device) 
        K = (c_list[idx][0].mm(K_rbf)).mm(c_list[j][0])
        
        mle_term = K.mm(log_prob_grad[j].unsqueeze(1)).squeeze()
        
        if grad is None:
            grad = mle_term.detach().clone()
        else:
            grad = grad + mle_term.detach().clone()
        
        for k1 in range(d):
            for k2 in range(d): 
                grad_k = torch.autograd.grad(K[k1, k2].sum(), inputs, allow_unused=True, retain_graph=True)[0]
                grad[k1] = grad[k1] + grad_k[j, k2]
    grad = grad - alpha * (c_list[idx][1] / c_list[idx][2].norm().pow(2)) * c_list[idx][2].squeeze()

    return grad
    
def get_gradient_fast(model, inputs, alpha):
    n = inputs.size(0)
    inputs = inputs.detach().requires_grad_(True)
    log_prob, constraint = model.get_log_prob_and_constraint(inputs)
    log_prob_grad = torch.autograd.grad(log_prob.sum(), inputs, allow_unused=True, retain_graph=True)[0]    
    constraint_grad = torch.autograd.grad(constraint.sum(), inputs, allow_unused=True, create_graph=True)[0]
    s_perp, g_para = project_g(log_prob_grad,  constraint_grad)
    rbf_kernel_matrix = kernel_rbf(inputs)
    
    svgd_gradient = torch.zeros_like(inputs, device=inputs.device)
    for i in range(n):
        svgd_gradient[i, :] = get_single_particle_gradient_with_rbf_and_c_fast(\
                                i, inputs, constraint, constraint_grad, rbf_kernel_matrix, s_perp,alpha).detach().clone()
        
    gradient = svgd_gradient / n
    
    return -gradient, log_prob.sum(), constraint.mean(), log_prob

def get_single_particle_gradient_with_rbf_and_c_fast(idx, inputs, constraint, constraint_grad, rbf_kernel_matrix, s_perp,alpha):
    n = inputs.size(0)
    d = inputs.shape[1]
    grad = None
    for j in range(n):
        mle_term, _ = project_g(s_perp[None,idx,:]*rbf_kernel_matrix[idx,j],  constraint_grad[None,j,:])
        
        if grad is None:
            grad = mle_term.detach().clone()
        else:
            grad = grad + mle_term.detach().clone()
        grad_k = torch.autograd.grad(rbf_kernel_matrix[idx,j], inputs, allow_unused=True, retain_graph=True)[0]
        temp, _ = project_g(grad_k[None,j,:],  constraint_grad[None,idx,:])
        dd_grad_k, _ = project_g(temp,  constraint_grad[None,j,:])
        grad += dd_grad_k
    grad -= alpha * constraint[idx]*constraint_grad[idx,:]/constraint_grad[idx,:].norm().pow(2)

    return grad

def median(tensor):
    """
    torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    tensor = tensor.detach().flatten()
    tensor_max = tensor.max()[None]
    return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.

def kernel_rbf(inputs):
    n = inputs.shape[0]
    pairwise_distance = torch.norm(inputs[:, None] - inputs, dim=2).pow(2)
    h = median(pairwise_distance) / math.log(n)
    kernel_matrix = torch.exp(-pairwise_distance / (1.*h+1e-6))

    return kernel_matrix

def project_g(v, dg):
    proj = torch.sum(v*dg,dim=1)/torch.sum(dg**2,dim=1)
    g_para =proj.unsqueeze(1).repeat(1,v.size(1))*dg
    g_perp = v - g_para
    return g_perp, g_para


def get_gradient_standard(model, inputs,alpha=100):
    n = inputs.size(0)
    inputs = inputs.detach().requires_grad_(True)
    log_prob, constraint = model.get_log_prob_and_constraint(inputs)
    log_prob_grad = torch.autograd.grad(log_prob.sum(), inputs, allow_unused=True, retain_graph=True)[0]    


    c_list = []
    for i in range(n):
        constraint_grad = torch.autograd.grad(constraint[i].sum(), inputs, allow_unused=True, create_graph=True)[0]
        constraint_grad = constraint_grad[i].unsqueeze(1)
        
        g_norm_sqr = constraint_grad.norm().pow(2)
        D = torch.eye(constraint_grad.shape[0],device=inputs.device) - constraint_grad@constraint_grad.t() / g_norm_sqr
        c_list.append((D, constraint[i], constraint_grad))
    
    rbf_kernel_matrix = kernel_rbf(inputs)
    
    svgd_gradient = torch.zeros_like(inputs, device=inputs.device)
    
    for i in range(n):
        svgd_gradient[i, :] = get_single_particle_gradient_with_rbf_and_c_standard(\
                            i, inputs, log_prob_grad, rbf_kernel_matrix, c_list, alpha).detach().clone()
    
    gradient = svgd_gradient / n

    
    return -gradient, log_prob.mean(), constraint.mean(), log_prob

def get_single_particle_gradient_with_rbf_and_c_standard(idx, inputs, log_prob_grad, rbf_kernel_matrix, c_list, alpha):
    n = inputs.size(0)
    d = inputs.shape[1]
    grad = None
    for j in range(n):        
        mle_term = rbf_kernel_matrix[idx, j]*log_prob_grad[j]
        
        if grad is None:
            grad = mle_term.detach().clone()
        else:
            grad = grad + mle_term.detach().clone()

        grad_k = torch.autograd.grad(rbf_kernel_matrix[idx, j], inputs, allow_unused=True, retain_graph=True)[0]
        grad += grad_k[j,:]
    return grad
 