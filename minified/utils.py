import torch
from PIL import Image as pillow
from transformers import AutoModelForCausalLM, AutoTokenizer

class config:
    lr = 2e-5
    data_split = 10_000
    img_size = 256
    patch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 2
    batch_size = 16

    # mdodel specs
    tr_blocks = 12
    freq_embed = 256
    attn_heads = 6
    hidden_size = 768

    # dataset/model ids
    dataset_id = "laion/laion2B-en-aesthetic"
    sd_vae_id = ''
    pretrained_id = ''
    moondream_id = "vikhyatk/moondream2"


moondream_model = AutoModelForCausalLM.from_pretrained(config.moondream_id, trust_remote_code=True)

md_tokenizer = AutoTokenizer.from_pretrained(config.moondream_id, trust_remote_code=True)


def caption_image(
    image: pillow,
    model = moondream_model,
    tokenizer = md_tokenizer,
) -> str:

    enc_image = model.encode_image(image)  # encode image with vision encoder(moondream uses SigLip)
    
    img_caption = model.answer_question(enc_image, "Describe this image and it's style", tokenizer)  # generate caption


    return img_caption