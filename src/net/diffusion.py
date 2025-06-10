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
            self.alpha = 1 - self.beta
            self.alphabar = torch.cumprod(self.alpha, dim=0)
        elif schedule == 'cosine':
            # Cosine noise schedule
            s = 0.008
            steps = T + 1
            x = torch.linspace(0, T, steps)
            alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.beta = torch.clip(betas, 0, 0.999)
            self.alpha = 1 - self.beta
            self.alphabar = torch.cumprod(self.alpha, dim=0)
            
        # Compute cumulative products
        self.betabar = torch.cumprod(self.beta, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphabar)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphabar)

   
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
        y = torch.zeros_like(x[:, 0:4, :, :]).to(device)
        
        # Calculate weights for mask averaging
        T_m = step_mask if step_mask < steps else steps
        w_p = (self.alphabar[:T_m]/torch.sum(self.alphabar[:T_m])).to(device)
        
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

    def dpm_solver_plus_plus_inverse(self, net, x, 
                                     intv=None, treat_cond=None, i_tg=None, 
                                     steps=None, step_mask=10, start_t=None,  device='cpu'):
        # TODO 
        # Inverse diffusion process for DPM-Solver++， not working well for now, need to fix some unknown bugs in the code.
        b, _, h, w = x.shape
        i_tg = i_tg if i_tg is not None else -torch.ones((b,), dtype=torch.int8)
        # i_tg = i_tg.to(torch.int64)
        steps = steps or self.T
        
        # Weight calculations
        T_m = min(step_mask, steps)
        w_p = self.alphabar[:T_m] / self.alphabar[:T_m].sum()
        
        y = torch.zeros_like(x[:, :4, :, :])
        times = torch.linspace(0, self.T - 1, steps).long().to(self.device).flip(0)
        
        for t, t_next in zip(times[:-1], times[1:]):
            with torch.no_grad():
                pred = net(x, t.repeat(b).to(self.device), intv_t=intv, treat_code=treat_cond, i_tg=i_tg)
                img_p, mask = pred[:, 4:7, :, :], pred[:, :4, :, :]
            
            
                        # Reshape x for easier indexing
            x = x.view(b, 4, 3, h, w).contiguous()
            
            # Extract and concatenate relevant slices of x
            xt = [x[[i], j, :, :, :] for i, j in zip(range(b), i_tg)]
            xt = torch.cat(xt, 0)
            # xt = x.view(b, 4, 3, h, w).gather(1, i_tg[:, None, None, None, None].expand(-1, -1, 3, h, w)).squeeze(1)
            
            lambda_t, lambda_t_next = torch.log(self.alphabar[t]), torch.log(self.alphabar[t_next])
            lambda_h = lambda_t - lambda_t_next
            
            D1 = (img_p - xt) / (1 - self.alphabar[t])
            if t_next >= 0:
                pred_next = net(x.view(b, 12, h, w), t_next.repeat(b).to(self.device), intv_t=intv, treat_code=treat_cond, i_tg=i_tg)
                img_p_next = pred_next[:, 4:7, :, :]
                D1_next = (img_p_next - xt) / (1 - self.alphabar[t_next])
                D2 = (D1_next - D1) / lambda_h
            else:
                D2 = torch.zeros_like(D1)
            
            x_next = xt + (1 - self.alphabar[t]) * (
                (1 - torch.exp(-lambda_h)) / lambda_h * D1 +
                (torch.exp(-lambda_h) + lambda_h - 1) / lambda_h ** 2 * D2
            )
            
            # Update x with new x0 estimate
            x = x.view(b, 4, 3, h, w).contiguous()
            for i, j in zip(range(b), i_tg):
                x[i, j, :, :, :] = x_next[i, :, :, :]
            
            # Reshape x back to original shape
            x = x.reshape(b, 12, h, w)
            # x.view(b, 4, 3, h, w).scatter_(1, i_tg[:, None, None, None, None].expand(-1, -1, 3, h, w), x_next.unsqueeze(1))
            
            if t <= T_m:
                y += mask * w_p[max(0, t-1)]
        
        return torch.tensor(x_next, device=self.device), torch.tensor(y, device=self.device)
    
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
        # cond_img = x[:, 0:9, :, :]
        
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
            
            # Reshape x for easier indexing with flexible i_tg indices
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
