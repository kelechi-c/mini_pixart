import torch
from datasets import load_dataset
from transformers import AutoTokenizer, T5EncoderModel
from torch.utils.data import IterableDataset, DataLoader
from .utils import config, caption_image
from katara import image_loader, read_image

# T5 text encoder
t5_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
t5_model = T5EncoderModel.from_pretrained("google-t5/t5-small")

def text_t5_encode(text_input: str, tokenizer=t5_tokenizer, model=t5_model):
    input_ids = tokenizer(text_input, return_tensors="pt").input_ids  # Batch size 1
    outputs = model(input_ids=input_ids)
    last_hidden_states = outputs.last_hidden_state

    return last_hidden_states


def preprocess_dataset(datapoint):
    pil_img = image_loader(datapoint["URL"]) # load to PIL format

    text_caption = caption_image(pil_img)
    datapoint['moondream_caption'] = text_caption
    datapoint['encoded_text'] = text_t5_encode(text_caption)

    return datapoint


hfdata = load_dataset(config.dataset_id, split='train', streaming=True, trust_remote_code=True).take(config.data_split)

hfdata = hfdata.map(preprocess_dataset, batched=True, batch_size=16) # will do this separately

class T2Idata(IterableDataset):
    def __init__(self, dataset=hfdata, config=config):
        super().__init__()
        self.dataset = dataset
        self.config = config
        
    def __len__(self):
        return self.config.data_split
    
    def __iter__(self):
        
        for sample in self.dataset:
            image = sample['image']
            image = read_image(image, config.img_size)
            image = torch.as_tensor(image, dtype=torch.float32)
            
            caption = torch.as_tensor(sample['encoded_text'], dtype=torch.float32)
            
            yield image, caption
            
t2i_dataset = T2Idata()
train_loader = DataLoader(t2i_dataset, batch_size=config.batch_size, num_workers=0)