"""

Added the off-the-shelf classifier guidance for DDPM sampling
"""

import enum
import math
import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from off_guided_diffusion.gaussian_diffusion import *
import torch
from off_guided_diffusion.respace import *


class GaussianDiffusionMLTCDiv2(SpacedDiffusion):

    def __init__(self,
        use_timesteps,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        super(GaussianDiffusionMLTCDiv2, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                   model_mean_type=model_mean_type,
                                                   model_var_type=model_var_type,
                                                   loss_type=loss_type,
                                                   rescale_timesteps=rescale_timesteps)

    def p_sample(
        self,
        model,
        x,
        t,
        gamma_factor=0.0,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Batch consider

        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        self.t = t

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        if cond_fn is not None:
            pred_xstart = out['pred_xstart']
            variance = out['variance'][0][0][0][0].item()
            self.max_variance = max(self.max_variance, variance)
            # adding sin timely-decay factor to the guidance schedule
            current_time = t[0].item()
            add_value = max(np.sin((current_time / self.num_timesteps) * np.pi) * self.max_variance * gamma_factor, 0.0)

            out["mean"], gradient_div = self.condition_mean(
                cond_fn, out, x, t, nonzero_mask, model_kwargs=model_kwargs
            )
        else:
            noise = th.randn_like(x)
            gradient_div = nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        sample = out["mean"] + gradient_div
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def condition_mean_mtl(self, cond_fn, p_mean_var, x, t, nonzero_mask, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn([x, p_mean_var['pred_xstart']], self._scale_timesteps(t), **model_kwargs)
        gradient_cls = p_mean_var["variance"] * gradient.float()
        gradient_gen = _extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * p_mean_var['pred_xstart']

        noise = th.randn_like(x)
        gradient_div = nonzero_mask * th.exp(0.5 * p_mean_var["log_variance"]) * noise

        g_cls_shape = gradient_cls.shape
        g_gen_shape = gradient_gen.shape
        assert g_cls_shape == g_gen_shape

        #diffusion - classification conflict
        new_gradient_cls = self.project_conflict(gradient_cls.clone(), gradient_gen.clone(), g_cls_shape) # check if gradient_cls change / changed -> using clone -> does not change
        new_gradient_gen = self.project_conflict(gradient_gen.clone(), gradient_cls.clone(), g_gen_shape) # check if gradient_gen change / changed -> using clone -> does not change

        # classification - diversity conflict
        final_gradient_cls = self.project_conflict(new_gradient_cls.clone(), gradient_div.clone(), g_cls_shape)
        final_gradient_div = self.project_conflict(gradient_div.clone(), new_gradient_cls.clone(), g_cls_shape)

        new_mean = (
            p_mean_var["mean"].float() - gradient_gen.float() + new_gradient_gen.float() + final_gradient_cls.float()
        )
        del gradient_cls, gradient_gen, new_gradient_gen, new_gradient_cls, final_gradient_cls
        return new_mean, final_gradient_div

    def condition_mean(self, cond_fn, *args, **kwargs):
        return self.condition_mean_mtl(self._wrap_model(cond_fn), *args, **kwargs)

    def project_conflict(self, grad1, grad2, shape):
        new_grad1 = torch.flatten(grad1, start_dim=1)
        new_grad2 = torch.flatten(grad2, start_dim=1)

        # g1 * g2 --------------- (batchsize,)
        g_1_g_2 = torch.sum(new_grad1 * new_grad2, dim=1)
        g_1_g_2 = torch.clamp(g_1_g_2, max=0.0)

        # ||g2||^2 ----------------- (batchsize,)
        norm_g2 = new_grad2.norm(dim=1) **2
        if torch.any(norm_g2 == 0.0):
            return new_grad1.view(shape)

        # (g1 * g2)/||g2||^2 ------------------- (batchsize,)
        g12_o_normg2 = g_1_g_2/norm_g2
        g12_o_normg2 = torch.unsqueeze(g12_o_normg2, dim=1)
        # why zero has problem?
        # g1
        new_grad1 -= ((g12_o_normg2) * new_grad2)
        new_grad1 = new_grad1.view(shape)
        return new_grad1

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
