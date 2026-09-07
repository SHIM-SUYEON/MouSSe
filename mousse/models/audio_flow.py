import math
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torchdiffeq import odeint

from .transformer import Transformer, ConvPositionEmbed
from .tensors import drop_using_mask, merge_mask

import torch.autograd.forward_ad as fwad

class UncertaintyHead(torch.nn.Module):
    """
    Uncertainty Head designed to predict the heteroscedastic velocity variance 
    over the exact same Time-Frequency resolution grid of the AudioFlow backbone.
    """
    def __init__(self, config):
        super(UncertaintyHead, self).__init__()
        self.config = config.model
        
        # Linear projection to ingest the concatenated state representation
        self.input_layer = torch.nn.Linear(2 * config.audio.n_mels, self.config.n_dim)
        self.sinu_pos_emb = LearnedSinusoidalPosEmb(self.config.n_dim)
        self.conv_embed = ConvPositionEmbed(n_dim = self.config.n_dim, kernel_size = 31)
        
        # Lightweight parallel Transformer blocks dedicated to variance modeling
        self.transformer = Transformer(
            n_heads = max(2, self.config.n_heads // 2), # Resource efficient scaling
            n_layers = max(2, self.config.n_layers // 2),
            n_dim = self.config.n_dim,
            n_dim_head = self.config.n_dim_head,
            n_dim_ffn = self.config.n_dim_ffn,
            n_non_bias_tokens = 1,
            att_dropout = 0.1,
            ffn_dropout = 0.1,
            cache_alibi = False
        )
        
        # Outputs localized log-variance matching the Mel-spectrogram channel dimension
        self.prediction = torch.nn.Linear(self.config.n_dim, config.audio.n_mels)

    def forward(self, *, audio, noise, times):
        # Concatenate audio features and current latent state
        output = torch.cat([audio, noise], dim = -1)
        output = self.input_layer(output)
        
        # Apply temporal positional scaling
        sinu_times = self.sinu_pos_emb(times).unsqueeze(1)
        output = torch.cat([output, sinu_times], dim = 1)
        output = self.conv_embed(output) + output
        
        # Process through the specialized uncertainty backbone
        output = self.transformer(output)
        output = self.prediction(output)
        
        # Slicing the boundary to synchronize time frame mapping
        return output[:, :-1, :]

class AudioFlow(torch.nn.Module):
    def __init__(self, config, *, cache_alibi = False):
        super(AudioFlow, self).__init__()
        self.config = config.model
        self.audio_config = config.audio

        # Transformer input
        self.transformer_input = torch.nn.Linear(2 * config.audio.n_mels, self.config.n_dim)

        # Sinusoidal positional embedding for time
        self.sinu_pos_emb = LearnedSinusoidalPosEmb(self.config.n_dim)

        # Convolutional positional encoder
        self.conv_embed = ConvPositionEmbed(n_dim = self.config.n_dim, kernel_size = 31)

        # Transformer
        self.transformer = Transformer(
            n_heads = self.config.n_heads,
            n_layers = self.config.n_layers,
            n_dim = self.config.n_dim,
            n_dim_head = self.config.n_dim_head,
            n_dim_ffn = self.config.n_dim_ffn,
            n_non_bias_tokens = 1, # Exclude time embedding from attention bias
            att_dropout = 0,
            ffn_dropout = 0.1,
            cache_alibi = cache_alibi
        )

        # Prediction
        self.prediction = torch.nn.Linear(self.config.n_dim, config.audio.n_mels)

    # def sample(self, *, audio, mask = None, steps, alpha = None, return_trajectory = False):
    @torch.inference_mode(False)
    def sample_with_uncertainty(self, *, audio, uq_head, steps, alpha = 1.5, lambda_max = 2.0):
        
        #
        # Prepare
        #
        device = audio.device
        batch_size, seq_len, n_mels = audio.shape[0], audio.shape[1], audio.shape[2]
        
        z = torch.randn_like(audio, device=device)
        U = torch.zeros_like(audio, device=device)
        
        dt = 1.0 / steps
        times = torch.linspace(0, 1, steps, device=device)
        
        for i in range(steps-1):
            t_curr = times[i]
            t_batch = fill_like_step = torch.full((batch_size,), t_curr, device=device, dtype=audio.dtype)
            
            audio_empty = torch.zeros_like(audio)
            audio_mix = torch.stack([audio_empty, audio], dim=0)
            z_mix = torch.stack([z, z], dim=0)
            # t_mix = torch.stack([t_batch, t_batch], dim=0)
            t_mix = torch.cat([t_batch, t_batch], dim=0)
            
            # Compute extrapolated velocity mean field
            with torch.no_grad():
                predicted_mix = self.forward(audio=audio_mix, noise=z_mix, times=t_mix)
                v_cond = predicted_mix[1]
                v_uncond = predicted_mix[0]
                
                v_guided = v_cond + (v_cond - v_uncond) * alpha
            
            # Compute instantaneous velocity variance
            with torch.no_grad():
                log_sigma_mix = uq_head(audio=audio_mix, noise=z_mix, times=t_mix)
                sigma_cond = torch.exp(log_sigma_mix[1])
                sigma_uncond = torch.exp(log_sigma_mix[0])
                
                numerator = torch.sum(sigma_uncond * sigma_cond - sigma_cond**2, dim=-1, keepdim=True)
                denominator = torch.sum((sigma_cond - sigma_uncond)**2, dim=-1, keepdim=True).clamp(min=1e-8)
                lambda_opt = torch.clamp(numerator / denominator, min=0.0)
                lambda_star = torch.clamp(lambda_opt, max=lambda_max)
                
                var_velocity = ((1.0 + lambda_star) * sigma_cond - lambda_star * sigma_uncond) ** 2
            
            # Compute covariance term via Autograd JVP
            rademacher = torch.randint(0, 2, z.shape, device=device, dtype=z.dtype) * 2 - 1
            std_state = torch.sqrt(U.clamp(min=1e-8))
            v_vector = std_state * rademacher
            
            # def local_velocity_fn(z_input):
            #     mix_in = torch.stack([audio_empty, audio], dim=0)
            #     z_in = torch.stack([z_input, z_input], dim=0)
            #     t_in = torch.stack([t_batch, t_batch], dim=0)
            #     out = self.forward(audio=mix_in, noise=z_in, times=t_in)
                
            #     out_guided = out[1]+(out[1]-out[0])*alpha
                
            #     return out_guided
            
            # def local_velocity_fn(z_input):
            #     mix_in = torch.stack([audio_empty, audio], dim=0)
            #     z_in = torch.stack([z_input, z_input], dim=0)
            #     # t_in = torch.full((2, batch_size), t_curr, device=device, dtype=audio.dtype)
            #     t_in = torch.full((2 * batch_size,), t_curr, device=device, dtype=audio.dtype)
                
            #     # Handle speaker identity conditions
            #     c_in = torch.stack([condition, condition], dim=0) if condition is not None else None
                
            #     out = self.forward(audio=mix_in, noise=z_in, condition=c_in, times=t_in)
            #     return out[1] + (out[1] - out[0]) * alpha
            
            def local_velocity_fn(z_input):
                audio_empty = torch.zeros_like(audio)
                t_in = torch.full((batch_size,), t_curr, device=device, dtype=audio.dtype)
                
                # Run cond and uncond separately — halves SDPA peak allocation vs stacking
                v_uncond = self.forward(audio=audio_empty, noise=z_input, times=t_in)
                v_cond   = self.forward(audio=audio,       noise=z_input, times=t_in)
                
                return v_cond + (v_cond - v_uncond) * alpha
            
            
            # Compute Jacobian-Vector Product tracking local spatial sensitivity
            # _, jvp_out = torch.autograd.functional.jvp(local_velocity_fn, (z,), (v_vector,))
            
            with fwad.dual_level():
                z_dual = fwad.make_dual(z, v_vector)
                out_dual = local_velocity_fn(z_dual)
                _, jvp_out = fwad.unpack_dual(out_dual)

            if jvp_out is None:
                jvp_out = torch.zeros_like(z)
            
            cov_term = v_vector * jvp_out

            # Accumulate state uncertainty matrix over Time-Frequency domain
            # Incorporates both instantaneous velocity variance and local error propagation covariance
            U = U + (var_velocity * dt) ** 2 + 2.0 * dt * cov_term

            # Update latent state space via euler forward integration
            z = z + v_guided * dt
            
        u_temporal = torch.mean(U, dim=-1) # Shape: [Batch, Time_Frames]

        return z, u_temporal
    
    def forward(self, *, audio, noise, condition = None, times, mask = None, target = None, mask_loss = False):
        if mask is None and target is not None and mask_loss:
            raise ValueError('Mask is required when target is provided and mask_loss enabled')
        if target is None and mask is not None:
            raise ValueError('Mask is not required when target is not provided')
        if condition is not None:
            assert condition.shape[0] == audio.shape[0]
            assert condition.shape[1] == audio.shape[1]
            assert condition.shape[2] == self.config.n_dim

        assert audio.shape[0] == noise.shape[0]
        assert audio.shape[1] == noise.shape[1]
        assert audio.shape[2] == noise.shape[2]

        # Concatenate spatial and masking state features
        output = torch.cat([audio, noise], dim = -1)
        output = self.transformer_input(output)

        if condition is not None:
            output = output + condition

        # sinu_times = self.sinu_pos_emb(times).unsqueeze(1)
        raw_sinu_times = self.sinu_pos_emb(times)
        if output.ndim == 4 and raw_sinu_times.ndim == 2:
            sinu_times = raw_sinu_times.unsqueeze(1).unsqueeze(1) # Align to 4D: [2, 1, 1, Features]
        else:
            sinu_times = raw_sinu_times.unsqueeze(1)
        output = torch.cat([output, sinu_times], dim=1)
        output = self.conv_embed(output) + output
        output = self.transformer(output)
        output = self.prediction(output)
        output = output[:, :-1, :]

        if target is not None:
            loss = F.mse_loss(output, target, reduction = 'none')
            loss = reduce(loss, 'b n d -> b n', 'mean')

            if mask_loss:
                loss = loss.masked_fill(~mask, 0.)
                n_masked_frames = mask.sum(dim = -1).clamp(min = 1)
                loss = loss.sum(dim = -1) / n_masked_frames
            else:
                loss = loss.sum(dim = -1) / target.shape[1]

            loss = loss.mean()
            return output, loss
        else:
            return output


class LearnedSinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        self.weights = torch.nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        # x = x.flatten()
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        return fouriered