import torch

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
    dataset_id = ''
    sd_vae_id = ''
    pretrained_id = ''