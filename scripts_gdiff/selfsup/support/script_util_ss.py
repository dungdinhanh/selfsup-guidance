from guided_diffusion.script_util import *
import torch.nn as nn
from guided_diffusion.unet import timestep_embedding
import torchvision.models as tmodels
import torch as th
import math
from evaluations.imagenet_evaluator_models.models_imp import *
from guided_diffusion.gaussian_diffusion_ss import *


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



class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, load_backbone=True):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=1000, pretrained=load_backbone)

        self.encoder.fc  = nn.Linear(2048, 2048)
        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer
        self.sampling = False

    def forward_2views(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

    def forward_1view(self, x1):
        z1 = self.encoder(x1)
        p1 = self.predictor(z1)
        return p1, z1

    def forward(self, x1, x2=None):
        if self.sampling:
            return self.forward_1view(x1)
        else:
            return self.forward_2views(x1, x2)


def get_simsiam_basemodel(base_model="resnet50"):
    if base_model == "resnet50":
        return resnet50
    elif base_model == "resnet18":
        return resnet18
    elif base_model == "resnet101":
        return resnet101
    elif base_model == "resnet152":
        return resnet152
    else:
        return resnet50


def create_simsiam_selfsup(dim, pred_dim, image_size, base_model="resnet50", load_backbone=True):
    # base_model = tmodels.__dict__['resnet50']
    base_model = get_simsiam_basemodel(base_model)
    model = SimSiam(base_model, dim, pred_dim, load_backbone=load_backbone)

    # if image_size == 64 or image_size == 128:
    #     # Change first layer
    #     conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
    #     n = conv1.kernel_size[0] * conv1.kernel_size[1] * conv1.out_channels
    #     conv1.weight.data.normal_(0, math.sqrt(2. / n))
    #     model.encoder.conv1 = conv1

    return model
    pass

def create_mocov2_selfsup(dim, pred_dim, image_size, base_model="resnet50", load_backbone=True):
    # base_model = tmodels.__dict__['resnet50']
    base_model = get_simsiam_basemodel(base_model)

    model = base_model(num_classes=128)
    dim_mlp = model.fc.weight.shape[1]
    model.fc = nn.Sequential(
        nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc
    )
    return model
    pass

def create_mocov3_selfsup(dim, pred_dim, image_size, base_model="resnet50", load_backbone=True):
    # base_model = tmodels.__dict__['resnet50']
    base_model = get_simsiam_basemodel(base_model)

    model = base_model(4096)
    hidden_dim = model.fc.weight.shape[1]
    model.fc = build_mlp(2, hidden_dim, 4096, 256)
    # dim_mlp = model.fc.weight.shape[1]
    # print(dim_mlp)
    # exit(0)
    # model.fc = nn.Sequential(
    #     nn.Linear(dim_mlp, dim_mlp*2), nn.ReLU(), model.fc
    # )
    return model
    pass

def build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)

def create_byol_selfsup(dim, pred_dim, image_size, base_model="resnet50", load_backbone=True):
    # base_model = tmodels.__dict__['resnet50']
    base_model = get_simsiam_basemodel(base_model)

    model = base_model(num_classes=128)
    dim_mlp = model.fc.weight.shape[1]
    model.fc = nn.Sequential(
        nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc
    )
    return model
    pass




def simsiam_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        image_size=64,
        dim=2048,
        pred_dim=512,
        base_model="resnet50"
    )

def simsiam_and_diffusion_defaults():
    res = simsiam_defaults()
    res.update(diffusion_defaults())
    return res


def create_simsiam_and_diffusion(
    image_size,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    pred_dim=512,
    dim=2048,
    base_model="resnet50"
):
    simsiam = create_simsiam_selfsup(dim, pred_dim, image_size, base_model)

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
    return simsiam, diffusion

def create_mocov2_and_diffusion(
    image_size,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    pred_dim=512,
    dim=2048,
    base_model="resnet50"
):
    simsiam = create_mocov2_selfsup(dim, pred_dim, image_size, base_model)

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
    return simsiam, diffusion

def create_mocov3_and_diffusion(
    image_size,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    pred_dim=512,
    dim=2048,
    base_model="resnet50"
):
    simsiam = create_mocov3_selfsup(dim, pred_dim, image_size, base_model)

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
    return simsiam, diffusion

def create_byol_and_diffusion(
    image_size,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    pred_dim=512,
    dim=2048,
    base_model="resnet50"
):
    simsiam = create_byol_selfsup(dim, pred_dim, image_size, base_model)

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
    return simsiam, diffusion

def create_model_and_diffusion_ss(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = create_gaussian_diffusion_ss(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_gaussian_diffusion_ss(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return GaussianDiffusionSS(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
