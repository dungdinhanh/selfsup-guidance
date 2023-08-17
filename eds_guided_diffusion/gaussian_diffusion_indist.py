"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

from eds_guided_diffusion.respace import *
import torch.nn as nn
from guided_diffusion.nn import GroupNorm32
import inspect


class InversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)


    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = th.norm(module.running_var.data - var, 2) + th.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


class InversionFeatureHookAutoLog():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.track_mean_xt = None
        self.track_var_xt = None
        self.track_xt = True


    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        N, C, H, W = input.size()
        G = 32
        assert C % G == 0

        # if self.track_xt and (self.track_mean_xt is not None or self.track_var_xt is not None):
        input = input.view(N, G, -1)
        mean = input.mean(-1, keepdim=True)
        var = input.var(-1, keepdim=True)

        if self.track_xt:
            if self.track_mean_xt is None and self.track_var_xt is None: # first case forward xt
                self.track_mean_xt = mean
                self.track_var_xt = var
                self.track_xt = False
            else: # third case forward xtg, but no tracking this is for final updating
                self.track_mean_xt = None
                self.track_var_xt = None
            r_feature = None
        else: # second case forward xtg, no tracking, but calculate loss
            r_feature = th.norm(self.track_mean_xt - mean, 2) + th.norm(self.track_var_xt - var, 2)
            self.track_xt = True
        self.r_feature = r_feature
        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        # r_feature = th.norm(module.running_var.data - var, 2) + th.norm(
        #     module.running_mean.data - mean, 2)

        # self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


class InversionFeatureHookTrack(InversionFeatureHook):
    track_xt_global = False
    calculate_r_loss = False

    def __init__(self, module):
        super(InversionFeatureHookTrack, self).__init__(module)
        self.track_mean_xt = None
        self.track_var_xt = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        input_shape = input[0].size()
        if len(input_shape) == 3:
            N, C, HW = input_shape
        else:
            N, C, H, W = input_shape
        G = 32
        assert C % G == 0

        # if self.track_xt and (self.track_mean_xt is not None or self.track_var_xt is not None):
        input = input[0].view(N, G, -1)
        mean = input.mean(-1, keepdim=True)
        var = input.var(-1, keepdim=True)

        if self.track_xt_global:
            self.track_mean_xt = mean.detach()
            self.track_var_xt = var.detach()
            print("____________________________________-")
            if self.track_mean_xt is None:
                print(1)
            print("______________________________________--")
            r_feature = None
        else: # second case forward xtg, no tracking, but calculate loss
            if self.calculate_r_loss:
                r_feature = th.norm(self.track_mean_xt - mean, 2) + th.norm(self.track_var_xt - var, 2)
            else:
                r_feature = None
        self.r_feature = r_feature


class GaussianDiffusionBNLoss(SpacedDiffusion):
    """
        Add bn loss
        Utilities for training and sampling diffusion models.

        Ported directly from here, and then adapted over time to further experimentation.
        https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

        :param betas: a 1-D numpy array of betas for each diffusion timestep,
                      starting at T and going to 1.
        :param model_mean_type: a ModelMeanType determining what the model outputs.
        :param model_var_type: a ModelVarType determining how variance is output.
        :param loss_type: a LossType determining the loss function to use.
        :param rescale_timesteps: if True, pass floating point timesteps into the
                                  model so that they are always scaled like in the
                                  original paper (0 to 1000).
        """

    def __init__(self,
                use_timesteps,
                 model,
                 **kwargs
                ):
        super(GaussianDiffusionBNLoss, self).__init__(use_timesteps=use_timesteps, **kwargs)
        self.model_hooks = None
        self.register_hook(model)

    def register_hook(self, diffusion_model):
        self.model_hooks = []
        for module in diffusion_model.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, GroupNorm32):
                self.model_hooks.append(InversionFeatureHook(module))

    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None
    ):
        """
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
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}


class GaussianDiffusionNormXt(SpacedDiffusion):
    """
        Add bn loss
        Utilities for training and sampling diffusion models.

        Ported directly from here, and then adapted over time to further experimentation.
        https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

        :param betas: a 1-D numpy array of betas for each diffusion timestep,
                      starting at T and going to 1.
        :param model_mean_type: a ModelMeanType determining what the model outputs.
        :param model_var_type: a ModelVarType determining how variance is output.
        :param loss_type: a LossType determining the loss function to use.
        :param rescale_timesteps: if True, pass floating point timesteps into the
                                  model so that they are always scaled like in the
                                  original paper (0 to 1000).
        """

    def __init__(self,
                use_timesteps,
                 model,
                 **kwargs
                ):
        super(GaussianDiffusionNormXt, self).__init__(use_timesteps=use_timesteps, **kwargs)


    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None
    ):
        """
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
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise + out["guidance"]
        if t[0] >=100:
            sample = (sample - out['mean'])/(th.exp(0.5 * out["log_variance"]))
        #     print(th.norm(sample))
        # print("________________________________________________________________")
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def condition_mean_guidance(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        guidance = cond_fn(x, self._scale_timesteps(t), **model_kwargs)

        # p_mean_var['mean'] = (
        #         p_mean_var["mean"].float() + p_mean_var["variance"] * guidance['scale'] * guidance['gradient'].float()
        # )
        p_mean_var['mean'] = p_mean_var['mean'].float()
        p_mean_var['guidance'] = p_mean_var["variance"] * guidance["scale"] * guidance["gradient"].float()
        return p_mean_var

    def condition_mean(self, cond_fn, *args, **kwargs):
        return self.condition_mean_guidance(self._wrap_model(cond_fn), *args, **kwargs)


class GaussianDiffusionNormXtGN(SpacedDiffusion):
    """
        Add bn loss
        Utilities for training and sampling diffusion models.

        Ported directly from here, and then adapted over time to further experimentation.
        https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

        :param betas: a 1-D numpy array of betas for each diffusion timestep,
                      starting at T and going to 1.
        :param model_mean_type: a ModelMeanType determining what the model outputs.
        :param model_var_type: a ModelVarType determining how variance is output.
        :param loss_type: a LossType determining the loss function to use.
        :param rescale_timesteps: if True, pass floating point timesteps into the
                                  model so that they are always scaled like in the
                                  original paper (0 to 1000).
        """

    def __init__(self,
                use_timesteps,
                 model,
                 first_gn_scale,
                 **kwargs
                ):
        super(GaussianDiffusionNormXtGN, self).__init__(use_timesteps=use_timesteps, **kwargs)
        self.model_hooks = None
        self.register_hook(model)
        self.first_gn_scale = first_gn_scale

    def register_hook(self, diffusion_model):
        self.model_hooks = []
        for module in diffusion_model.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, GroupNorm32):
                self.model_hooks.append(InversionFeatureHookTrack(module))

    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None
    ):
        """
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
        # first step sampling
        InversionFeatureHookTrack.track_xt_global = False
        InversionFeatureHookTrack.calculate_r_loss = False
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        if t[0] != 0:
            # forward xt
            InversionFeatureHookTrack.track_xt_global = True
            InversionFeatureHookTrack.calculate_r_loss = False
            sample = out['mean'] + nonzero_mask * th.exp(0.5 * out['log_variance']) * noise
            _ = self.p_mean_variance(
                model,
                sample,
                t - 1,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )

            # forward xt guidance
            InversionFeatureHookTrack.track_xt_global = False
            InversionFeatureHookTrack.calculate_r_loss = True
            with th.enable_grad():
                sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise + out["guidance"]

                sample_grad = sample.detach().requires_grad_(True)

                _ = self.p_mean_variance(
                    model,
                    sample_grad,
                    t-1,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                rescale = [self.first_gn_scale] + [1. for _ in range(len(self.model_hooks) - 1)]
                loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.model_hooks)])
                print(loss_r_feature)
                r_features_grad = th.autograd.grad(loss_r_feature, sample_grad)[0]
                print(r_features_grad)
            sample = sample - r_features_grad
        else:
            sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise + out["guidance"]
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def condition_mean_guidance(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        guidance = cond_fn(x, self._scale_timesteps(t), **model_kwargs)

        # p_mean_var['mean'] = (
        #         p_mean_var["mean"].float() + p_mean_var["variance"] * guidance['scale'] * guidance['gradient'].float()
        # )
        p_mean_var['mean'] = p_mean_var['mean'].float()
        p_mean_var['guidance'] = p_mean_var["variance"] * guidance["scale"] * guidance["gradient"].float()
        return p_mean_var

    def condition_mean(self, cond_fn, *args, **kwargs):
        return self.condition_mean_guidance(self._wrap_model(cond_fn), *args, **kwargs)





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
