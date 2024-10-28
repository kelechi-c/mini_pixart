from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from .utils import config, caption_image
from katara import image_loader, read_image


def caption_dataset(datapoint):
    img_url = image_loader(datapoint["URL"]) # load to PIL format
    img = read_image(img_url, config.img_size) # load to numpy array

    datapoint['moondream_caption'] = caption_image(img)

    return datapoint


hfdata = load_dataset(config.dataset_id, split='train', streaming=True, trust_remote_code=True).take(config.data_split)

data = hfdata.map(caption_dataset, batched=True, batch_size=16)