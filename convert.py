import torch

dt = torch.load("ldm_1p4b_init0.ckpt", map_location="cpu")
unet = {}
vqvae = {}
ldmbert = {}
for k,v in dt['state_dict'].items():
    print(k,v.shape)
    
    unet_key = "model.diffusion_model."
    if unet_key in k:
        unet[k.replace(unet_key, "")] = v

    vqvae_key = "first_stage_model."
    if vqvae_key in k:
        vqvae[k.replace(vqvae_key, "")] = v

    ldmbert_key = "cond_stage_model."
    if ldmbert_key in k:
        ldmbert[k.replace(ldmbert_key, "")] = v
        
import os
os.makedirs("init_weights")
torch.save(unet, "init_weights/unet.pt")
torch.save(vqvae, "init_weights/vqvae.pt")
torch.save(ldmbert, "init_weights/ldmbert.pt")
