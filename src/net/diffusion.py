import numpy as np
import torch
import math

class GaussianDiffusion():
    '''Gaussian Diffusion process with linear beta scheduling'''
    def __init__(self, T, schedule):
        # Number of diffusion steps
        self.T = T
    
        # Initialize noise schedule
        if schedule == 'linear':
            # Linear noise schedule
            b0=1e-4
            bT=2e-2
            self.beta = np.linspace(b0, bT, T)
        elif schedule == 'cosine':
            # Cosine noise schedule
            self.alphabar = self.__cos_noise(np.arange(0, T+1, 1)) / self.__cos_noise(0) # Generate an extra alpha for bT
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)
            
        # Compute cumulative products
        self.betabar = np.cumprod(self.beta)
        self.alpha = 1 - self.beta
        self.alphabar = np.cumprod(self.alpha)

    def __cos_noise(self, t):
        # Helper function for cosine noise schedule
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t/self.T + offset) / (1+offset)) ** 2
   
    def sample(self, x0, t):        
        # Sample from the diffusion process at time t
        noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))        
        atbar = torch.from_numpy(np.asarray(self.alphabar[t-1])).view(noise_dims).to(x0.device)
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
        w_p = self.alphabar[:T_m]/self.alphabar[:T_m].sum()
        
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
            x0 = (1 / np.sqrt(at)) * (xt.cpu().numpy()  - ((1-at) / np.sqrt(1-atbar)) * img_p.cpu().numpy()) + np.sqrt(beta_tilde) * z.cpu().numpy()
            x0 = torch.tensor(x0).to(device)

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
