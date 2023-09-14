from guided_diffusion.vision_images import *
from PIL import ImageFilter

class ImageNetHF2Views(ImageNet):
    def __init__(self, resolution, random_crop=False, random_flip=True, split='train', classes=True, miniset=False):
        # to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        super(ImageNetHF2Views, self).__init__(split=split, transform=None, check_data=True, miniset=miniset)
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.local_classes = classes
        augmentation = [
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
        ]
        self.transform = transforms.Compose(augmentation)


    def __getitem__(self, indices):
        imgs_bytes = self.reader.read(indices)
        samples = []
        for i, bytes_ in enumerate(imgs_bytes):
            img = pickle.loads(bytes_).convert("RGB")
            label = self.meta["targets"][indices[i]]
            samples.append((img, int(label)))

        transformed_samples = []
        for img, label in samples:
            img1 = self.transform(img)
            img2 = self.transform(img)
            if self.random_crop:
                arr1 = random_crop_arr(img1, self.resolution)
                arr2 = random_crop_arr(img2, self.resolution)
            else:
                arr1 = center_crop_arr(img1, self.resolution)
                arr2 = center_crop_arr(img2, self.resolution)

            if self.random_flip and random.random() < 0.5:
                arr1 = arr1[:, ::-1]
                arr2 = arr2[:, ::-1]

            img1 = arr1.astype(np.float32) / 127.5 - 1
            img1 = np.transpose(img1, [2, 0, 1]) 

            img2 = arr2.astype(np.float32) / 127.5 - 1
            img2 = np.transpose(img2, [2, 0, 1])
            # if self.transform:
            #     img = self.transform(img)
            out_dict = {}
            if self.local_classes:
                out_dict["y"] = label

            transformed_samples.append((img1, img2, out_dict))
        return transformed_samples

class ImageNetHF3Views2Imgs(ImageNet):
    def __init__(self, resolution, random_crop=False, random_flip=True, split='train', classes=True, miniset=False):
        # to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        super(ImageNetHF3Views2Imgs, self).__init__(split=split, transform=None, check_data=True, miniset=miniset)
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.local_classes = classes
        augmentation = [
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
        ]
        self.transform = transforms.Compose(augmentation)

    def _getnegative(self, indices):
        new_indices = np.random.randint(len(self), size=len(indices))
        compare = (new_indices == indices)
        new_indices[compare] += 1
        larger_than_len = (new_indices >= len(self))
        new_indices[larger_than_len] -= 2


        imgs_bytes = self.reader.read(new_indices)
        samples = []
        for i, bytes_ in enumerate(imgs_bytes):
            img = pickle.loads(bytes_).convert("RGB")
            label = self.meta["targets"][indices[i]]
            samples.append((img, int(label)))

        transformed_samples = []
        for img, label in samples:
            img = self.transform(img)
            if self.random_crop:
                arr = random_crop_arr(img, self.resolution)
            else:
                arr = center_crop_arr(img, self.resolution)

            if self.random_flip and random.random() < 0.5:
                arr = arr[:, ::-1]

            img = arr.astype(np.float32) / 127.5 - 1
            img = np.transpose(img, [2, 0, 1])  # might not need to transpose
            # if self.transform:
            #     img = self.transform(img)
            out_dict = {}
            if self.local_classes:
                out_dict["y"] = label

            transformed_samples.append((img, out_dict))
        return transformed_samples


    def __getitem__(self, indices):
        imgs_bytes = self.reader.read(indices)
        samples = []
        for i, bytes_ in enumerate(imgs_bytes):
            img = pickle.loads(bytes_).convert("RGB")
            label = self.meta["targets"][indices[i]]
            samples.append((img, int(label)))

        transformed_samples = []
        transformed_samples3 = self._getnegative(indices)

        for indx, (img, label) in enumerate(samples):
            img1 = self.transform(img)
            img2 = self.transform(img)
            if self.random_crop:
                arr1 = random_crop_arr(img1, self.resolution)
                arr2 = random_crop_arr(img2, self.resolution)
            else:
                arr1 = center_crop_arr(img1, self.resolution)
                arr2 = center_crop_arr(img2, self.resolution)

            if self.random_flip and random.random() < 0.5:
                arr1 = arr1[:, ::-1]
                arr2 = arr2[:, ::-1]

            img1 = arr1.astype(np.float32) / 127.5 - 1
            img1 = np.transpose(img1, [2, 0, 1])

            img2 = arr2.astype(np.float32) / 127.5 - 1
            img2 = np.transpose(img2, [2, 0, 1])
            # if self.transform:
            #     img = self.transform(img)
            out_dict = {}
            if self.local_classes:
                out_dict["y"] = label

            transformed_samples.append((img1, img2, transformed_samples3[indx][0], out_dict, transformed_samples3[indx][1]))
        return transformed_samples

class ImageNetHFAug(ImageNet):
    def __init__(self, resolution, random_crop=False, random_flip=True, split='train', classes=True, miniset=False):
        # to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        super(ImageNetHFAug, self).__init__(split=split, transform=None, check_data=True, miniset=miniset)
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.local_classes = classes
        augmentation = [
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
        ]
        self.transform = transforms.Compose(augmentation)

    def __getitem__(self, indices):
        imgs_bytes = self.reader.read(indices)
        samples = []
        for i, bytes_ in enumerate(imgs_bytes):
            img = pickle.loads(bytes_).convert("RGB")
            label = self.meta["targets"][indices[i]]
            samples.append((img, int(label)))

        transformed_samples = []
        for img, label in samples:
            img = self.transform(img)
            if self.random_crop:
                arr = random_crop_arr(img, self.resolution)
            else:
                arr = center_crop_arr(img, self.resolution)

            if self.random_flip and random.random() < 0.5:
                arr = arr[:, ::-1]

            img = arr.astype(np.float32) / 127.5 - 1
            img = np.transpose(img, [2, 0, 1]) # might not need to transpose
            # if self.transform:
            #     img = self.transform(img)
            out_dict = {}
            if self.local_classes:
                out_dict["y"] = label

            transformed_samples.append((img, out_dict))
        return transformed_samples


class ImageNetHFAug_la(ImageNetHFAug):
    def __init__(self, resolution, random_crop=False, random_flip=True, split='train', classes=True, miniset=False):
        super(ImageNetHFAug_la, self).__init__(resolution, random_crop, random_flip, split, classes, miniset)
        if split == "train":
            augmentation = [
                transforms.RandomHorizontalFlip(),
            ]
            self.transform = transforms.Compose(augmentation)
        else:
            self.transform = transforms.Compose([])


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

