import numpy as np 
import torch
import torch.nn as nn
import torch.distributions as dists
import math

class O_Langevin(nn.Module):
    def __init__(self, logp, g, stepsize=1e-1, alpha = 1, beta=1, useHessian=True, M = 1000):
        super(O_Langevin,self).__init__()
        self.logp = logp
        self.g = g
        self.stepsize = stepsize
        self.alpha = alpha
        self.beta = beta
        self.useHessian = useHessian
        self.M = M 
        self.dim = 2

    def step(self, x):
        xi = torch.randn_like(x)
        Dlogpx = self.compute_grad(x,self.logp)
        v = self.stepsize*Dlogpx + (2*self.stepsize)**.5*xi
        gx = self.g(x)
        Dgx = self.compute_grad(x,self.g)   
        g_perp, g_para = self.project_g(v,  Dgx)

        phi = self.alpha*torch.sign(gx)*torch.abs(gx)**self.beta        
        Dgx2 = torch.sum(Dgx**2,dim=1,keepdim=True).repeat(1,self.dim)
        if self.useHessian==True: 
            DxD = torch.zeros_like(x)
            # calculating the Hessian term. 
            for j in range(x.shape[0]):
                term1 = torch.sum(Dgx[j,:]**2)
                tDgx = Dgx[j,:]/term1
                Hgx = self.compute_hessian(x[[j],:],self.g)
                term3 = 2*torch.sum(tDgx @ Hgx * tDgx) * Dgx[j,:]
                DxD[j,:] = tDgx @ Hgx  +  tDgx * torch.trace(Hgx) - term3
            dx = g_perp - self.stepsize*phi.unsqueeze(1).repeat(1,self.dim)*Dgx/Dgx2 - self.stepsize*DxD
        else:
            dx =  g_perp - self.stepsize*phi.unsqueeze(1).repeat(1,self.dim)*Dgx/Dgx2

        x = x +  torch.clip(dx, -self.M, self.M)    
        return x

    def compute_grad(self,x, model):
        x = x.requires_grad_()
        gx = torch.autograd.grad(model(x).sum(), x)[0]
        return gx.detach()

    def compute_hessian(self,x, model):
        x = x.requires_grad_()
        Hgx = torch.autograd.functional.hessian(model, x).squeeze()
        return Hgx.detach()


    def project_g(self,v, dg):
        proj = torch.sum(v*dg,dim=1)/torch.sum(dg**2,dim=1)
        g_para =proj.unsqueeze(1).repeat(1,self.dim)*dg
        g_perp = v - g_para
        return g_perp, g_para


class O_SVGD(nn.Module):
    def __init__(self, logp, g, stepsize=1e-1, alpha = 20., M = 1000):
        super(O_SVGD,self).__init__()
        self.logp = logp
        self.g = g
        self.stepsize = stepsize
        self.alpha = alpha
        self.M = M

    def step(self, particle):
        grad, c = self.svgd_get_gradient(self.logp, self.g, particle)
        dx = grad.detach().clone() * self.stepsize
        particle.data = particle.data +  torch.clip(dx, -self.M, self.M)### negative sign for constraint SVGD
        return particle

    def get_single_particle_gradient_with_rbf_and_c(self,idx, inputs, log_prob_grad, rbf_kernel_matrix, c_list):
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

        grad_final = grad - self.alpha * (c_list[idx][1] / c_list[idx][2].norm().pow(2)) * c_list[idx][2].squeeze()
        return grad

    def median(self,tensor):
        """
        torch.median() acts differently from np.median(). We want to simulate numpy implementation.
        """
        tensor = tensor.detach().flatten()
        tensor_max = tensor.max()[None]
        return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.

    def kernel_rbf(self,inputs):
        n = inputs.shape[0]
        pairwise_distance = torch.norm(inputs[:, None] - inputs, dim=2).pow(2)
        h = self.median(pairwise_distance) / math.log(n)
        kernel_matrix = torch.exp(-pairwise_distance / (1.*h+1e-6))

        return kernel_matrix

    def svgd_get_gradient(self,model, constraint, inputs):
        n = inputs.size(0)
        inputs = inputs.detach().requires_grad_(True)

        log_prob = model(inputs)
        log_prob_grad = torch.autograd.grad(log_prob.sum(), inputs, allow_unused=True, retain_graph=True)[0]

        constraint_value = constraint(inputs)

        c_list = []
        for i in range(n):
            constraint_grad = torch.autograd.grad(constraint_value[i].sum(), inputs, allow_unused=True, create_graph=True)[0]
            constraint_grad = constraint_grad[i].unsqueeze(1)
            
            g_norm_sqr = constraint_grad.norm().pow(2)
            D = torch.eye(constraint_grad.shape[0]) - constraint_grad@constraint_grad.t() / g_norm_sqr
            c_list.append((D, constraint_value[i], constraint_grad))
        
        rbf_kernel_matrix = self.kernel_rbf(inputs)
        
        svgd_gradient = torch.zeros_like(inputs, device=inputs.device)

        for i in range(n):
            svgd_gradient[i, :] = self.get_single_particle_gradient_with_rbf_and_c(\
                                i, inputs, log_prob_grad, rbf_kernel_matrix, c_list).detach().clone()
        
        gradient = svgd_gradient / n

        return gradient.squeeze(), constraint_value

    def svgd_get_gradient_fast(self,model, constraint, inputs):
        n = inputs.size(0)
        inputs = inputs.detach().requires_grad_(True)

        log_prob = model(inputs)
        log_prob_grad = torch.autograd.grad(log_prob.sum(), inputs, allow_unused=True, retain_graph=True)[0]
        constraint_value = constraint(inputs)
        constraint_grad = torch.autograd.grad(constraint_value.sum(), inputs, allow_unused=True, create_graph=True)[0]

        s_perp, g_para = self.project_g(log_prob_grad,  constraint_grad)
        rbf_kernel_matrix = self.kernel_rbf(inputs)

        svgd_gradient = torch.zeros_like(inputs, device=inputs.device)
        for i in range(n):
            svgd_gradient[i, :] = self.get_single_particle_gradient_with_rbf_and_c_fast(\
                                    i, inputs, constraint_value, constraint_grad, rbf_kernel_matrix, s_perp).detach().clone()
        
        gradient = svgd_gradient / n
        return gradient.squeeze(), constraint_value

    def get_single_particle_gradient_with_rbf_and_c_fast(self,idx, inputs, constraint, constraint_grad, rbf_kernel_matrix, s_perp):
        n = inputs.size(0)
        d = inputs.shape[1]
        grad = None
        for j in range(n):
            mle_term, _ = self.project_g(s_perp[None,idx,:]*rbf_kernel_matrix[idx,j],  constraint_grad[None,j,:])
            
            if grad is None:
                grad = mle_term.detach().clone()
            else:
                grad = grad + mle_term.detach().clone()
            grad_k = torch.autograd.grad(rbf_kernel_matrix[idx,j], inputs, allow_unused=True, retain_graph=True)[0]
            
            temp, _ = self.project_g(grad_k[None,j,:],  constraint_grad[None,idx,:])
            dd_grad_k, _ = self.project_g(temp,  constraint_grad[None,j,:])
            grad += dd_grad_k
        grad -= self.alpha * constraint[idx]*constraint_grad[idx,:]/constraint_grad[idx,:].norm().pow(2)

        return grad
    def project_g(self,v, dg):
        proj = torch.sum(v*dg,dim=1)/torch.sum(dg**2,dim=1)
        g_para =proj.unsqueeze(1).repeat(1,v.size(1))*dg
        g_perp = v - g_para
        return g_perp, g_para

class GD(nn.Module):
    def __init__(self, logp, g, stepsize=1e-1):
        super(GD,self).__init__()
        self.logp = logp
        self.g = g
        self.stepsize = stepsize
        self.dim = 2

    def step(self, x):
        gx = self.g(x)
        Dgx = torch.sign(gx)*self.compute_grad(x,self.g)   
        x = x - self.stepsize*Dgx 
        return x

    def compute_grad(self,x, model):
        x = x.requires_grad_()
        gx = torch.autograd.grad(model(x).sum(), x)[0]
        return gx.detach()