from guided_diffusion.resample import *


def create_named_schedule_sampler_ext(name, diffusion, distance=10, p = 700):
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
    elif name == "uniform-2-steps":
        return UniformSampler2steps(diffusion, distance)
    elif name == "uniform-2-steps-weight":
        return UniformSampler2stepsWeights(diffusion, distance)
    elif name == "uniform-2-steps-control":
        return UniformSampler2stepsControl(diffusion)
    elif name == "uniform-2-steps-control-max600":
        return UniformSampler2stepsControlMax600(diffusion)
    elif name == "uniform-2-steps-control-maxp":
        return UniformSampler2stepsControlMaxP(diffusion, p=p)
    elif name == "uniform-2-steps-control-maxp-wl":
        return UniformSampler2stepsControlMaxPWL(diffusion, p=p)
    elif name == "uniform-2-steps-control-maxp-wl2":
        return UniformSampler2stepsControlMaxPWL2(diffusion, p=p)
    elif name == "uniform-2-steps-control-maxp-wl3":
        return UniformSampler2stepsControlMaxPWL3(diffusion, p=p)
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

class UniformSampler2steps(ScheduleSampler):
    def __init__(self, diffusion, distance=10):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])
        self.distance = distance

    def weights(self):
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
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indx_distance = np.random.choice(self.distance, size=(batch_size,))
        indices_np2 = indices_np + indx_distance
        indices_np2 = np.clip(indices_np2, 0, len(p) - 1)
        indices1 = th.from_numpy(indices_np).long().to(device)
        indices2 = th.from_numpy(indices_np2).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices1, indices2, weights


class UniformSampler2stepsWeights(UniformSampler2steps):
    def __init__(self, diffusion, distance=10, ):
        super().__init__(diffusion, distance)

    def weights(self):
        pass

class UniformSampler2stepsControl(UniformSampler2steps):
    def __init__(self, diffusion, distance=[]):
        super().__init__(diffusion)

        if len(distance) == 0:
            self.distance = [(400, 500), (20, 50), (10, 20)]
        else:
            self.distance = []
            for i in range(0, len(distance), step=2):
                self.distance.append((distance[i], distance[i+1]))

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
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)

        indx_distance = np.zeros_like(indices_np)

        # range 1: > 600
        items_range1 = indices_np >= 600
        indx_distance[items_range1] = np.random.random_integers(self.distance[0][0], self.distance[0][1], indx_distance[items_range1].shape)

        items_range2 = ((indices_np < 600) & (indices_np >= 300))
        indx_distance[items_range2] = np.random.random_integers(self.distance[1][0], self.distance[1][1], indx_distance[items_range2].shape)

        items_range3 = indices_np < 300
        indx_distance[items_range3] = np.random.random_integers(self.distance[2][0], self.distance[2][1], indx_distance[items_range3].shape)

        # indx_distance = np.random.choice(self.distance, size=(batch_size,))

        indices_np2 = indices_np - indx_distance
        indices_np2 = np.clip(indices_np2, 0, len(p) - 1)
        indices1 = th.from_numpy(indices_np).long().to(device)
        indices2 = th.from_numpy(indices_np2).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices1, indices2, weights


class UniformSampler2stepsControlMax600(UniformSampler2stepsControl):
    def __init__(self, diffusion, distance=[]):
        super().__init__(diffusion)


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
        max_p = len(p) - 300
        indices_np = np.random.choice(max_p, size=(batch_size,), p=p)

        indx_distance = np.zeros_like(indices_np)

        # range 1: > 600
        items_range1 = indices_np >= 600
        indx_distance[items_range1] = np.random.random_integers(self.distance[0][0], self.distance[0][1], indx_distance[items_range1].shape)

        items_range2 = ((indices_np < 600) & (indices_np >= 300))
        indx_distance[items_range2] = np.random.random_integers(self.distance[1][0], self.distance[1][1], indx_distance[items_range2].shape)

        items_range3 = indices_np < 300
        indx_distance[items_range3] = np.random.random_integers(self.distance[2][0], self.distance[2][1], indx_distance[items_range3].shape)

        # indx_distance = np.random.choice(self.distance, size=(batch_size,))

        indices_np2 = indices_np - indx_distance
        indices_np2 = np.clip(indices_np2, 0, len(p) - 1)
        indices1 = th.from_numpy(indices_np).long().to(device)
        indices2 = th.from_numpy(indices_np2).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices1, indices2, weights

class UniformSampler2stepsControlMaxP(UniformSampler2stepsControl):
    def __init__(self, diffusion, p, distance=[]):
        super().__init__(diffusion)
        self.p = p


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
        max_p = self.p
        indices_np = np.random.choice(max_p, size=(batch_size,))

        indx_distance = np.zeros_like(indices_np)

        # range 1: > 600
        items_range1 = indices_np >= 600
        indx_distance[items_range1] = np.random.random_integers(self.distance[0][0], self.distance[0][1], indx_distance[items_range1].shape)

        items_range2 = ((indices_np < 600) & (indices_np >= 300))
        indx_distance[items_range2] = np.random.random_integers(self.distance[1][0], self.distance[1][1], indx_distance[items_range2].shape)

        items_range3 = indices_np < 300
        indx_distance[items_range3] = np.random.random_integers(self.distance[2][0], self.distance[2][1], indx_distance[items_range3].shape)

        # indx_distance = np.random.choice(self.distance, size=(batch_size,))

        indices_np2 = indices_np - indx_distance
        indices_np2 = np.clip(indices_np2, 0, len(p) - 1)
        indices1 = th.from_numpy(indices_np).long().to(device)
        indices2 = th.from_numpy(indices_np2).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices1, indices2, weights


class UniformSampler2stepsControlMaxPWL(UniformSampler2stepsControlMaxP):
    def __init__(self, diffusion, p, distance=[]):
        super().__init__(diffusion, p)
        # self.p = p


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
        max_p = self.p
        indices_np = np.random.choice(max_p, size=(batch_size,))
        weight_ng = np.ones_like(indices_np).astype(float)
        weight_pos = np.ones_like(indices_np).astype(float)

        indx_distance = np.zeros_like(indices_np)

        # range 1: > 600
        items_range1 = indices_np >= 600
        indx_distance[items_range1] = np.random.random_integers(self.distance[0][0], self.distance[0][1], indx_distance[items_range1].shape)
        weight_ng[items_range1] *= 1.0
        weight_pos[items_range1] *= 0.1


        items_range2 = ((indices_np < 600) & (indices_np >= 300))
        indx_distance[items_range2] = np.random.random_integers(self.distance[1][0], self.distance[1][1], indx_distance[items_range2].shape)
        weight_ng[items_range2] *= 0.5
        weight_pos[items_range2] *= 0.5

        items_range3 = indices_np < 300
        indx_distance[items_range3] = np.random.random_integers(self.distance[2][0], self.distance[2][1], indx_distance[items_range3].shape)
        weight_ng[items_range3] *= 0.0
        weight_pos[items_range3] *= 0.5

        items_range4 = indices_np < 100
        weight_pos[items_range4] = 1.0

        # indx_distance = np.random.choice(self.distance, size=(batch_size,))

        indices_np2 = indices_np - indx_distance
        indices_np2 = np.clip(indices_np2, 0, len(p) - 1)
        indices1 = th.from_numpy(indices_np).long().to(device)
        indices2 = th.from_numpy(indices_np2).long().to(device)
        # weights_np = 1 / (len(p) * p[indices_np])
        # weights = th.from_numpy(weights_np).float().to(device)
        weight_ng = th.from_numpy(weight_ng).float().to(device)
        weight_pos = th.from_numpy(weight_pos).float().to(device)
        return indices1, indices2, weight_ng, weight_pos


class UniformSampler2stepsControlMaxPWL2(UniformSampler2stepsControlMaxP):
    def __init__(self, diffusion, p, distance=[]):
        super().__init__(diffusion, p)
        # self.p = p


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
        max_p = self.p
        indices_np = np.random.choice(max_p, size=(batch_size,))
        weight_ng = np.ones_like(indices_np).astype(float)
        weight_pos = np.ones_like(indices_np).astype(float)

        indx_distance = np.zeros_like(indices_np)

        # range 1: > 600
        items_range1 = indices_np >= 600
        indx_distance[items_range1] = np.random.random_integers(self.distance[0][0], self.distance[0][1], indx_distance[items_range1].shape)
        weight_pos[items_range1] *= 0.001
        weight_ng[items_range1] *= 0.001


        items_range2 = ((indices_np < 600) & (indices_np >= 100))
        indx_distance[items_range2] = np.random.random_integers(self.distance[1][0], self.distance[1][1], indx_distance[items_range2].shape)
        weight_pos[items_range2] *= 0.1
        weight_ng[items_range2] *= 0.1

        items_range3 = indices_np < 100
        indx_distance[items_range3] = np.random.random_integers(self.distance[2][0], self.distance[2][1], indx_distance[items_range3].shape)
        weight_pos[items_range3] *= 1.0
        weight_ng[items_range3] *= 1.0


        # indx_distance = np.random.choice(self.distance, size=(batch_size,))

        indices_np2 = indices_np - indx_distance
        indices_np2 = np.clip(indices_np2, 0, len(p) - 1)
        indices1 = th.from_numpy(indices_np).long().to(device)
        indices2 = th.from_numpy(indices_np2).long().to(device)
        # weights_np = 1 / (len(p) * p[indices_np])
        # weights = th.from_numpy(weights_np).float().to(device)
        weight_ng = th.from_numpy(weight_ng).float().to(device)
        weight_pos = th.from_numpy(weight_pos).float().to(device)
        return indices1, indices2, weight_ng, weight_pos

class UniformSampler2stepsControlMaxPWL3(UniformSampler2stepsControlMaxP):
    def __init__(self, diffusion, p, distance=[]):
        super().__init__(diffusion, p)
        # self.p = p


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
        max_p = self.p
        indices_np = np.random.choice(max_p, size=(batch_size,))
        weight_ng = np.ones_like(indices_np).astype(float)
        weight_pos = np.ones_like(indices_np).astype(float)

        indx_distance = np.zeros_like(indices_np)

        # range 1: > 600
        items_range1 = indices_np >= 600
        indx_distance[items_range1] = np.random.random_integers(self.distance[0][0], self.distance[0][1], indx_distance[items_range1].shape)
        weight_pos[items_range1] *= 0.001


        items_range2 = ((indices_np < 600) & (indices_np >= 100))
        indx_distance[items_range2] = np.random.random_integers(self.distance[1][0], self.distance[1][1], indx_distance[items_range2].shape)
        weight_pos[items_range2] *= 0.1

        items_range3 = indices_np < 100
        indx_distance[items_range3] = np.random.random_integers(self.distance[2][0], self.distance[2][1], indx_distance[items_range3].shape)
        weight_pos[items_range3] *= 1.0


        # indx_distance = np.random.choice(self.distance, size=(batch_size,))

        indices_np2 = indices_np - indx_distance
        indices_np2 = np.clip(indices_np2, 0, len(p) - 1)
        indices1 = th.from_numpy(indices_np).long().to(device)
        indices2 = th.from_numpy(indices_np2).long().to(device)
        # weights_np = 1 / (len(p) * p[indices_np])
        # weights = th.from_numpy(weights_np).float().to(device)
        weight_ng = th.from_numpy(weight_ng).float().to(device)
        weight_pos = th.from_numpy(weight_pos).float().to(device)
        return indices1, indices2, weight_ng, weight_pos




