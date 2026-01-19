import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

device = "cuda"

model_path = "GonzaloMG/marigold-normals"

tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(device)
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(device)

unet.eval()
text_encoder.eval()

with torch.no_grad():
    B, H, W = 1, 64, 64
    latents = torch.randn((B, 4, H, W), device=device)
    noise = torch.randn_like(latents)
    unet_input = torch.cat([latents, noise], dim=1)

    timesteps = torch.tensor([999], device=device)

    empty_token = tokenizer(
        [""], padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to(device)

    encoder_hidden_states = text_encoder(empty_token)[0]

    print("UNet forward start")
    out = unet(unet_input, timesteps, encoder_hidden_states)[0]
    print("UNet forward done:", out.shape)
