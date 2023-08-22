from guided_diffusion.script_util import *
import torch.nn as nn
from .unet import timestep_embedding
import torchvision.models
import torch as th


def create_classifier_selfsup(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
        out_channels=2048,
        num_head_channels=64,
        pred_dim=512
):
    in_channels = 3
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    elif image_size == 28:
        channel_mult = (1, 2, 2)
        in_channels = 1
        out_channels = 512
        num_head_channels = 8
    else:
        raise ValueError(f"unsupported image size: {image_size}")
    dim = out_channels
    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUnetSSModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=classifier_width,
        out_channels=dim,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=num_head_channels,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
        pred_dim=pred_dim
    )

    # prev_dim = base_model._feature_size
    #
    # base_model.pool = (nn.Linear(prev_dim, prev_dim, bias=False),
    #                    nn.BatchNorm1d(prev_dim),
    #                    nn.ReLU(inplace=True),  # first layer
    #                    nn.Linear(prev_dim, prev_dim, bias=False),
    #                    nn.BatchNorm1d(prev_dim),
    #                    nn.ReLU(inplace=True),  # second layer
    #                    self.encoder.fc,  # need something here
    #                    nn.BatchNorm1d(dim, affine=False))  # output layer

def create_classifier_selfsup_direction(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
        out_channels=2048 + 128,
        num_head_channels=64,
        dim=2048,
        pred_dim=512
):
    in_channels = 3
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    elif image_size == 28:
        channel_mult = (1, 2, 2)
        in_channels = 1
        out_channels = 512
        num_head_channels = 8
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUnetSSModelDirection(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=classifier_width,
        out_channels=out_channels,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=num_head_channels,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
        dim=dim,
        pred_dim=pred_dim
    )


class EncoderUnetSSModel(EncoderUNetModel):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            pool="adaptive",
            pred_dim=512,

    ):
        super(EncoderUnetSSModel, self).__init__(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            pool=pool
        )
        dim = out_channels
        self.out = nn.Sequential(self.out,
                                 nn.Linear(dim, dim, bias=False),
                                 nn.BatchNorm1d(dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(dim, dim, bias=False),
                                 nn.BatchNorm1d(dim, affine=False),
                                 )

        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(pred_dim, dim))

        self.pred_prev = None
        self.z_prev = None
        self.release_z_grad = False

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            # if self.pool.startswith("spatial"):
            #     results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        # if self.pool.startswith("spatial"):
        #     results.append(h.type(x.dtype).mean(dim=(2, 3)))
        #     h = th.cat(results, axis=-1)
        #     return self.out(h)
        # else:
        h = h.type(x.dtype)
        # return self.out(h)
        z = self.out(h)
        p = self.predictor(z)
        if not self.release_z_grad:
            return p, z.detach()
        else:
            return p, z


class EncoderUnetSSModelDirection(EncoderUNetModel):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            pool="adaptive",
            dim=2048,
            pred_dim=512,

    ):
        super(EncoderUnetSSModelDirection, self).__init__(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            pool=pool
        )
        self.time_embedding_dim = out_channels - dim
        self.encoder = nn.Sequential(
                                     nn.Linear(dim, dim, bias=False),
                                     nn.BatchNorm1d(dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(dim, dim, bias=False),
                                     nn.BatchNorm1d(dim, affine=False),
        )

        self.time_predictor = nn.Sequential(
            nn.Linear(self.time_embedding_dim, 128, bias=False),
            nn.BatchNorm1d(self.time_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(pred_dim, dim))

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            # if self.pool.startswith("spatial"):
            #     results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        # if self.pool.startswith("spatial"):
        #     results.append(h.type(x.dtype).mean(dim=(2, 3)))
        #     h = th.cat(results, axis=-1)
        #     return self.out(h)
        # else:
        h = h.type(x.dtype)
        # return self.out(h)
        complex_latent = self.out(h)
        time_pred = self.time_predictor(complex_latent[:, :self.time_embedding_dim])
        z = self.encoder(complex_latent[:, self.time_embedding_dim:])
        p = self.predictor(z)
        return p, z.detach(), time_pred

def create_ssclassifier_and_diffusion_cifar10(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    pred_dim=512,
    out_channels=1024,
):
    classifier = create_classifier_selfsup(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
        out_channels=out_channels,
        pred_dim=pred_dim
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return classifier, diffusion

def create_ssclassifier_and_diffusion(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    pred_dim=512,
    out_channels=2048,
):
    classifier = create_classifier_selfsup(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
        out_channels=out_channels,
        pred_dim=pred_dim
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return classifier, diffusion


def create_ssclassifier_and_diffusion_direction(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    pred_dim=512,
    out_channels=2048 + 128,
    dim=2048,
):
    classifier = create_classifier_selfsup_direction(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
        out_channels=out_channels,
        dim=dim,
        pred_dim=pred_dim
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return classifier, diffusion



