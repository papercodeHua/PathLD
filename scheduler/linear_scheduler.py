import torch
from utils.config import config
from tqdm import tqdm

class Diffusion:
    """
    Diffusion Process (DDPM & DDIM).
    """
    def __init__(self, noise_steps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.0200):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.beta = self.prepare_noise_schedule().to(config.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) # alpha_bar
        
        self.alpha_bar = torch.cat(
            [torch.ones(1, device=self.alpha_hat.device), self.alpha_hat],
            dim=0
        ) 

    def prepare_noise_schedule(self) -> torch.Tensor:
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Forward Diffusion Process (q(x_t | x_0)).
        z_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n: int) -> torch.Tensor:
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, y, label=None, ages=None, saliency=None):
        """
        DDPM Sampling (Standard sampling).
        """
        T = self.noise_steps
        n = y.shape[0]
        x = torch.randn((n, 1, 40, 48, 40), device=config.device)

        with torch.no_grad():
            for t in range(T, 0, -1):
                t_batch = torch.full((n,), t, dtype=torch.long, device=config.device)

                ab_t    = self.alpha_bar[t].view(1,1,1,1,1).expand(n,1,1,1,1)
                ab_tm1  = self.alpha_bar[t-1].view(1,1,1,1,1).expand(n,1,1,1,1)
                a_t     = self.alpha[t-1].view(1,1,1,1,1).expand(n,1,1,1,1)
                b_t     = self.beta[t-1].view(1,1,1,1,1).expand(n,1,1,1,1)

                # Predict x0 
                x0 = model(x, y, t_batch, label=label, ages=ages, saliency=saliency)

                # Posterior mean 
                coef_x0 = torch.sqrt(ab_tm1) * b_t / (1.0 - ab_t + 1e-12)
                coef_xt = torch.sqrt(a_t) * (1.0 - ab_tm1) / (1.0 - ab_t + 1e-12)
                mean = coef_x0 * x0 + coef_xt * x

                # Posterior variance 
                var  = b_t * (1.0 - ab_tm1) / (1.0 - ab_t + 1e-12)
                
                if t > 1:
                    x = mean + torch.sqrt(var) * torch.randn_like(x)
                else:
                    x = mean
            return x

    def sample_ddim(self, model, y, label=None, ages=None, saliency=None, num_steps=100, seed=None, eta=0.0):
        """
        DDIM Sampling (Accelerated deterministic/stochastic sampling).
        """
        if seed is not None:
            torch.manual_seed(seed)
        device = config.device
        T = self.noise_steps
        n = y.shape[0]
        x = torch.randn((n, 1, 40, 48, 40), device=device)

        all_t = torch.linspace(T, 1, steps=num_steps, dtype=torch.long, device=device)

        with torch.no_grad():
            for i in range(len(all_t) - 1):
                t = all_t[i]          #
                t_prev = all_t[i + 1] 

                ab_t    = self.alpha_bar[t].view(1,1,1,1,1).expand(n,1,1,1,1)
                ab_prev = self.alpha_bar[t_prev - 1].view(1,1,1,1,1).expand(n,1,1,1,1)

                # Predict x0 (Signal-Reconstruction Parameterization)
                t_batch = t.view(1).expand(n)
                x0 = model(x, y, t_batch, label=label, ages=ages, saliency=saliency)

                eps_hat = (x - torch.sqrt(ab_t) * x0) / torch.sqrt(1.0 - ab_t + 1e-12)

                # DDIM sigma
                sigma_t = eta * torch.sqrt(
                    (1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_prev + 1e-12)
                )

                z = torch.randn_like(x) if (eta > 0.0 and t_prev.item() > 1) else torch.zeros_like(x)

                # DDIM Update Step 
                dir_coef = torch.sqrt(torch.clamp(1.0 - ab_prev - sigma_t**2, min=0.0))
                x = torch.sqrt(ab_prev) * x0 + dir_coef * eps_hat + sigma_t * z

            return x