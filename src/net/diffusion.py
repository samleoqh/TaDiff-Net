import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional

class GaussianDiffusion():
    '''Gaussian Diffusion process with linear beta scheduling'''
    def __init__(self, T, schedule, device='cpu'):
        # Number of diffusion steps
        self.T = T
        self.device = device
    
        # Initialize noise schedule
        if schedule == 'linear':
            # Linear noise schedule
            b0=1e-4
            bT=2e-2
            self.beta = torch.linspace(b0, bT, T, device=self.device)
        elif schedule == 'cosine':
            # Cosine noise schedule
            t = torch.arange(0, T+1, 1, device=self.device)
            self.alphabar = self.__cos_noise(t) / self.__cos_noise(torch.tensor(0, device=self.device))
            self.beta = torch.clamp(1 - (self.alphabar[1:] / self.alphabar[:-1]), max=0.999)
            
        # Compute cumulative products
        self.betabar = torch.cumprod(self.beta, dim=0)
        self.alpha = 1 - self.beta
        self.alphabar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphabar)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphabar)


    def __cos_noise(self, t):
        # Helper function for cosine noise schedule
        offset = 0.008
        return torch.cos(torch.tensor(np.pi * 0.5, device=self.device) *
                        (t/self.T + offset) / (1+offset)) ** 2
   
    def sample(self, x0, t):
        # Sample from the diffusion process at time t
        noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))
        atbar = self.alphabar[t-1].view(noise_dims).to(x0.device)
        assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'
        
        # Add noise to x0
        epsilon = torch.randn_like(x0)
        xt = torch.sqrt(atbar) * x0 + torch.sqrt(1-atbar) * epsilon
        return xt, epsilon

    def TaDiff_inverse(self, net, x=None, intv=None, treat_cond=None, i_tg= None,
                       steps=None, start_t=None, step_mask = 10, device='cpu'):
        # Inverse diffusion process for TaDiff

        # Set default values for start_t and steps
        start_t = int(start_t) if start_t else self.T
        steps = int(steps) if steps else self.T
        
        b, sc, h, w = x.shape  # b: batch size, sc: number of channels, h: height, w: width
        
        # Initialize target indices if not provided
        if i_tg is None:
            i_tg = -torch.ones((x.shape[0],), dtype=torch.int8)
        else:
            i_tg = i_tg
        
        # Extract the last 3 channels as x0
        x0 = x[:, 9:12, :, :]
        
        # Initialize output tensor for mask
        y = torch.zeros_like(x[:, 0:4, :, :])
        
        # Calculate weights for mask averaging
        T_m = step_mask if step_mask < steps else steps
        w_p = self.alphabar[:T_m]/torch.sum(self.alphabar[:T_m])
        
        # Inverse diffusion loop
        for t in range(start_t, start_t-steps, -1):
            at = self.alpha[t-1]
            atbar = self.alphabar[t-1]
            
            # Prepare noise for next step
            if t > 1:
                z = torch.randn_like(x0)
                atbar_prev = self.alphabar[t-2]
                beta_tilde = self.beta[t-1] * (1 - atbar_prev) / (1 - atbar)
            else:
                z = torch.zeros_like(x0)
                beta_tilde = 0

            # Generate prediction using the network
            with torch.no_grad():
                t = torch.tensor([t]).view(1)
                pred = net(x, t.float().to(device), intv_t=intv,treat_code=treat_cond, i_tg=i_tg)
                img_p, mask = pred[:, 4:7, :, :], pred[:, 0:4, :, :]
            
            # Reshape x for easier indexing
            x = x.view(b, 4, 3, h, w).contiguous()

            # Extract and concatenate relevant slices of x
            xt = [x[[i], j, :, :, :] for i, j in zip(range(b), i_tg)]
            xt = torch.cat(xt, 0)

            # Compute x0 estimate
            at_tensor = torch.tensor(at, device=device)
            beta_tilde_tensor = torch.tensor(beta_tilde, device=device)
            x0 = (1 / torch.sqrt(at_tensor)) * (xt - ((1-at_tensor) / torch.sqrt(1-atbar)) * img_p) + torch.sqrt(beta_tilde_tensor) * z

            # Update x with new x0 estimate
            for i, j in zip(range(b), i_tg):
                x[i, j, :, :, :] = x0[i, :, :, :]

            # Reshape x back to original shape
            x = x.reshape(b, 12, h, w)
            
            # Update mask prediction
            if steps > T_m:
                if t <= T_m: # only merge the last T_m masks predicted by the model
                    y += mask * w_p[t-1]
            else:
                y += mask * w_p[t-1]
    
        return x0, y

    # def ddim_inverse(self, net, x, eta=0.0, steps=50, device='cpu'):
    def ddim_inverse(self, net, x=None, intv=None, treat_cond=None, i_tg= None,
                       steps=None, start_t=None, step_mask = 10, eta=0., device='cpu'):
        """
        DDIM inverse process with fusion mechanism
        Args:
            net: The neural network model
            x: Input tensor (batch_size, 12, H, W)
            intv: Intervention tensor
            treat_cond: Treatment condition tensor
            i_tg: Target indices
            steps: Number of inverse steps
            start_t: Starting time step, not used in this function
            step_mask: Number of steps to use for mask averaging
            eta: Stochasticity parameter (0 for deterministic)
            device: Device to run on
        Returns:
            Reconstructed x0 and mask predictions
        """
        b, _, h, w = x.shape
        
        # Create time steps for inverse process
        times = torch.linspace(0, self.T-1, steps+1).long().to(device)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        # Initialize mask accumulator
        y = torch.zeros_like(x[:, 0:4, :, :])
        # x0 = x[:, 9:12, :, :]
        
        # Extract conditional image if  i_tg is 3, meaning allways predict future images
        # cond_img = x[:, 9:12, :, :]
        
        # Initialize target indices if not provided
        if i_tg is None:
            i_tg = -torch.ones((x.shape[0],), dtype=torch.int8)
        else:
            i_tg = i_tg
            
        # Calculate weights for mask averaging
        T_m = step_mask if step_mask < steps else steps
        w_p = self.alphabar[:T_m]/self.alphabar[:T_m].sum()
        
            
        # Inverse process loop
        for t, t_next in time_pairs:
            with torch.no_grad():
                pred = net(x, torch.full((b,), t, device=device, dtype=torch.long),
                           intv_t=intv,treat_code=treat_cond, i_tg=i_tg)
                img_p, mask = pred[:, 4:7, :, :], pred[:, 0:4, :, :]
            # print(f"Predicted img_p shape: {img_p.shape}, mask shape: {mask.shape}")
            
            # Reshape x for easier indexing
            x = x.view(b, 4, 3, h, w).contiguous()

            # Extract and concatenate relevant slices of x
            xt = [x[[i], j, :, :, :] for i, j in zip(range(b), i_tg)]
            xt = torch.cat(xt, 0)
            
            # Compute predicted x0
            sqrt_recip_alphas_cumprod_t = 1. / torch.sqrt(self.alphabar[t])
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            x0_pred = sqrt_recip_alphas_cumprod_t * xt - sqrt_one_minus_alphas_cumprod_t * img_p
            if t_next < 0:
                x_next = x0_pred
            else:
                alphas_cumprod_t = self.alphabar[t]
                alphas_cumprod_t_next = self.alphabar[t_next] if t_next >= 0 else torch.tensor(1.0)
                
                sigma = eta * torch.sqrt((1 - alphas_cumprod_t_next) / (1 - alphas_cumprod_t)) * \
                        torch.sqrt(1 - alphas_cumprod_t / alphas_cumprod_t_next)
                
                noise = torch.randn_like(x0_pred)
                x_next = torch.sqrt(alphas_cumprod_t_next) * x0_pred + \
                        torch.sqrt(1 - alphas_cumprod_t_next - sigma**2) * noise + \
                        sigma * noise
        
            # Update x with new x0 estimate
            for i, j in zip(range(b), i_tg):
                x[i, j, :, :, :] = x_next[i, :, :, :]
            
            # Reshape x back to original shape
            x = x.reshape(b, 12, h, w)
            
            if steps > T_m:
                if t <= T_m: # only merge the last T_m masks predicted by the model
                    y += mask * w_p[t-1]
                    # print(f"Added mask weight: {w_p[t-1]}")
            else:
                y += mask * w_p[t-1]

        
        return x_next, y
