from guided_diffusion.train_util import *


def log_loss_dict_nomean(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv(key, values.item())
        # Log the quantiles (four quartiles, in particular).
        # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #     quartile = int(4 * sub_t / diffusion.num_timesteps)
        #     logger.logkv(f"{key}_q{quartile}", sub_loss)