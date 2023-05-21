import numpy as np 
import torch
import torch.nn as nn
import torch.distributions as dists
import math

class O_Langevin(nn.Module):
    def __init__(self, model, alpha = 1, beta=1, wd=5e-4, datasize=50000):
        super(O_Langevin,self).__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.wd = wd 
        self.temperature = 1.
        self.datasize = datasize

    def step(self, floss, gloss,lr,epoch):
        Dlogpx = torch.autograd.grad(floss, self.model.parameters(),retain_graph=True)
        Dgx = torch.autograd.grad(gloss, self.model.parameters())
        phi = self.alpha*torch.sign(gloss)
        Dgnorm2 = sum([torch.sum(Dgxi**2) for Dgxi in Dgx])
        for Dlogpxi, Dgxi, theta in zip(Dlogpx,Dgx,self.model.parameters()):
            Dlogpxi.add_(theta.data, alpha=self.wd)
            if epoch<150:
                Dgxi.add_(theta.data, alpha=self.wd)
            eps = torch.randn_like(Dlogpxi)
            xi = (2.*lr*self.temperature/self.datasize)**.5*eps
            v = lr*Dlogpxi + xi
            g_perp, g_para = self.project_g(v,  Dgxi)           
            dx =  g_perp - lr*phi*Dgxi
            theta.data.add_(torch.clip(dx, -1000, 1000))  

    def project_g(self, v, dg):
        g_para =torch.sum(v*dg)/torch.sum(dg**2)*dg
        g_perp = v - g_para
        return g_perp, g_para


class O_SVGD(nn.Module):
    def __init__(self, model,particles, alpha = 1, beta=1, wd=5e-4, datasize=50000,num_particles=4,criterion=None,num_classes=10):
        super(O_SVGD,self).__init__()
        self.model = model
        self.particles = particles
        self.alpha = alpha
        self.beta = beta
        self.wd = wd 
        self.temperature = 1.
        self.datasize = datasize
        self.M=1000
        self.num_particles=num_particles
        self.param_shapes = []
        self.weighs_split = []
        for theta in model.parameters():
            self.param_shapes.append(theta.shape)
            self.weighs_split.append(np.prod(theta.shape))
        self.criterion = criterion
        self.num_classes = num_classes

    def median(self,tensor):
        """
        torch.median() acts differently from np.median(). We want to simulate numpy implementation.
        """
        tensor = tensor.detach().flatten()
        tensor_max = tensor.max()[None]
        return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.

    def kernel_rbf(self,inputs):
        inputs = inputs.detach().requires_grad_(True)
        n = inputs.shape[0]
        pairwise_distance = torch.norm(inputs[:, None] - inputs, dim=2).pow(2)
        h = self.median(pairwise_distance) / math.log(n)
        kernel_matrix = torch.exp(-pairwise_distance / (1.*h+1e-6))

        return kernel_matrix, inputs

    def forward_single(self,data,targets,i,evaluation=False):
        if evaluation:
            self.reshape_particles(self.particles[i]) 
            outputs = self.model(data)
            floss = self.criterion(outputs, targets)
            return floss,outputs
        self.model.zero_grad()
        self.reshape_particles(self.particles[i])         
        self.model.zero_grad()   
        outputs = self.model(data)
        floss = self.criterion(outputs, targets)
        gloss = floss
        return floss, gloss, outputs

    def reshape_particles(self,z):
        z_splitted = torch.split(z, self.weighs_split, 0)
        i=0
        for shape,theta in zip(self.param_shapes,self.model.parameters()):
            theta.data = z_splitted[i].reshape(shape)
            i+=1

    def step(self,data,targets, lr,epoch):
        log_prob_grad = torch.zeros_like(self.particles, device=self.particles.device)
        constraint_grad = torch.zeros_like(self.particles, device=self.particles.device)
        floss = torch.zeros(self.num_particles)
        gloss = torch.zeros(self.num_particles)
        outputs_list = []
        for i in range(self.num_particles):
            flossi,glossi,outputs = self.forward_single(data,targets,i)
            outputs_list.append(outputs)
            floss[i] = flossi
            gloss[i] = glossi
            temp_grad = torch.autograd.grad(flossi, self.model.parameters()) 
            log_prob_grad[i,:] = torch.cat([p.detach().data.clone().flatten() for p in temp_grad])  
            constraint_grad[i,:] = log_prob_grad[i,:].clone()
            log_prob_grad[i,:].add_(self.particles[i], alpha=self.wd)
            if epoch<150:
                constraint_grad[i,:].add_(self.particles[i], alpha=self.wd)
            del temp_grad
        if epoch<150:
            s_perp = None
        else:
            s_perp, g_para = self.project_g(log_prob_grad,  constraint_grad)
        rbf_kernel_matrix, particles_for_kernel = self.kernel_rbf(self.particles)
        
        svgd_gradient = torch.zeros_like(self.particles, device=self.particles.device)
        for i in range(self.num_particles):
            svgd_gradient[i, :] = self.get_single_particle_gradient_with_rbf_and_c_fast(\
                                    i, particles_for_kernel,gloss, constraint_grad, rbf_kernel_matrix, s_perp,epoch).detach().clone()
            
        self.particles.add_(torch.clip(lr*svgd_gradient, -1000, 1000))
        return floss.mean(), gloss.mean(),sum(outputs_list)/self.num_particles

    def get_single_particle_gradient_with_rbf_and_c_fast(self,idx, inputs, constraint, constraint_grad, rbf_kernel_matrix, s_perp,epoch):
        n = inputs.size(0)
        d = inputs.shape[1]
        grad = torch.zeros(constraint_grad.shape[1],device=constraint_grad.device)
        for j in range(n):
            if epoch>=150:
                mle_term, _ = self.project_g(s_perp[None,idx,:]*rbf_kernel_matrix[idx,j],  constraint_grad[None,j,:])
                grad = grad + mle_term.detach().clone()
            grad_k = torch.autograd.grad(rbf_kernel_matrix[idx,j], inputs, allow_unused=True, retain_graph=True)[0]
            temp, _ = self.project_g(grad_k[None,j,:],  constraint_grad[None,idx,:])
            dd_grad_k, _ = self.project_g(temp,  constraint_grad[None,j,:])
            grad += dd_grad_k.squeeze()
        grad /= n
        grad -= self.alpha * torch.sign(constraint[idx])*constraint_grad[idx,:]
        return grad

    def project_g(self,v, dg):
        proj = torch.sum(v*dg,dim=1)/torch.sum(dg**2,dim=1)
        g_para =proj.unsqueeze(1).repeat(1,v.size(1))*dg
        g_perp = v - g_para
        return g_perp, g_para

    def eval(self,data,targets,evaluation=True):
        outputs_list = []
        for i in range(self.num_particles):
            _,outputs = self.forward_single(data,targets,i,evaluation)
            outputs_list.append(outputs)       
        return sum(outputs_list)/self.num_particles

