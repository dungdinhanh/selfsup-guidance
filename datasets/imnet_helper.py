import hfai
import hfai.datasets

hfai.datasets.set_data_dir('/ssd/datasets/dungda/data/imagenet/')
hfai.datasets.download("ImageNet", miniset=False)