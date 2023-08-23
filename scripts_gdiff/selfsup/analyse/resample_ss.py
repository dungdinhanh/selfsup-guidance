from guided_diffusion.resample import *


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    elif name == "uniform-with-range":
        return UniformSamplerwRange(diffusion)
    elif name == "uniform-with-fix":
        return UniformSamplerwFix(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")

class UniformSamplerwRange(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])
        self.range = None

    def weights(self):
        self._weights = np.ones([self.range[1] - self.range[0] + 1])
        return self._weights

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(np.arange(self.range[0], self.range[1]+1), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        # weights_np = 1 / (len(p) * p[indices_np])
        # weights = th.from_numpy(weights_np).float().to(device)
        weights = None
        return indices, weights


class UniformSamplerwFix(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])
        self.range = None

    def weights(self):
        self._weights = np.ones([2])
        return self._weights

    def sample(self, batch_size, device, value):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        indices_np = np.asarray([value]*batch_size)
        indices = th.from_numpy(indices_np).long().to(device)
        weights = None
        return indices, weights