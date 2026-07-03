"""Self-contained Difix loader + single forward probe.

Loads the published diffusers-format weights from nvidia/difix directly with
diffusers' own classes (no external module-level code executed). Applies the
VAE skip-connection surgery + PEFT LoRA adapter so the published VAE state dict
(which stores PEFT-style base_layer / lora_A.vae_skip / lora_B.vae_skip keys)
loads cleanly. Runs ONE single-step forward on a dummy 512x512 render.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from peft import LoraConfig
from safetensors.torch import load_file
from transformers import AutoTokenizer, CLIPTextModel

REPO = "nvidia/difix"
DEVICE = "cuda:0"
DTYPE = torch.float32  # NVIDIA tests in FP32


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def build_difix():
    # tokenizer + text encoder from the difix repo's own copies (sd-turbo CLIP)
    tok = AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer")
    txt = CLIPTextModel.from_pretrained(REPO, subfolder="text_encoder").to(DEVICE, DTYPE)

    sched = DDPMScheduler.from_pretrained(REPO, subfolder="scheduler")
    sched.set_timesteps(1, device=DEVICE)
    sched.alphas_cumprod = sched.alphas_cumprod.to(DEVICE)

    # UNet: published weights
    unet = UNet2DConditionModel.from_pretrained(REPO, subfolder="unet").to(DEVICE, DTYPE)

    # VAE: build from CONFIG only (no weight load yet), do skip surgery + PEFT
    # adapter so the module's keys match the published PEFT-style checkpoint,
    # then load the published weights exactly once.
    vae_cfg_path = hf_hub_download(REPO, "vae/config.json")
    import json as _json
    vcfg = _json.load(open(vae_cfg_path))
    vcfg.pop("gamma", None)
    vcfg.pop("ignore_skip", None)
    vcfg.pop("lora_rank", None)
    vae = AutoencoderKL.from_config(vcfg)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, 1, 1, bias=False)
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, 1, 1, bias=False)
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, 1, 1, bias=False)
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, 1, 1, bias=False)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1

    # Attach the same LoRA adapter the checkpoint was trained with so PEFT keys exist.
    target_modules_vae = [
        "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
        "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
        "to_k", "to_q", "to_v", "to_out.0",
    ]
    targets = []
    for name, _ in vae.named_modules():
        if "decoder" in name and any(name.endswith(x) for x in target_modules_vae):
            targets.append(name)
    vae.add_adapter(
        LoraConfig(r=4, init_lora_weights="gaussian", target_modules=targets),
        adapter_name="vae_skip",
    )

    # Load published VAE weights (PEFT-style keys). assign=True handles any
    # meta-tensor params left by from_pretrained.
    vae_path = hf_hub_download(REPO, "vae/diffusion_pytorch_model.safetensors")
    sd = load_file(vae_path)
    res = vae.load_state_dict(sd, strict=False, assign=True)
    print(f"VAE load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    if res.missing_keys:
        print("  sample missing:", res.missing_keys[:5])
    vae = vae.to(DEVICE, DTYPE)

    for m in (txt, unet, vae):
        m.eval()
        m.requires_grad_(False)
    return tok, txt, sched, unet, vae


@torch.no_grad()
def difix_fix(render_uint8: np.ndarray, tok, txt, sched, unet, vae, prompt: str = "") -> np.ndarray:
    """(H,W,3) uint8 degraded render -> (H,W,3) uint8 cleaned image, single step."""
    h, w = render_uint8.shape[:2]
    nh, nw = h - h % 8, w - w % 8
    x = torch.from_numpy(render_uint8[:nh, :nw].astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0).to(DEVICE, DTYPE)
    x = x * 2.0 - 1.0  # [-1,1]

    ids = tok(prompt, max_length=tok.model_max_length, padding="max_length",
              truncation=True, return_tensors="pt").input_ids.to(DEVICE)
    enc = txt(ids)[0]

    # Single-step restore: encode -> predict epsilon at fixed t -> x0 directly.
    t = torch.tensor([199], device=DEVICE).long()
    z = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
    pred = unet(z, t, encoder_hidden_states=enc).sample
    acp = sched.alphas_cumprod[t].view(-1, 1, 1, 1)
    z_den = (z - (1 - acp).sqrt() * pred) / acp.sqrt()  # predicted x0
    vae.decoder.incoming_skip_acts = vae.encoder.current_down_blocks
    out = vae.decode(z_den / vae.config.scaling_factor).sample.clamp(-1, 1)
    out = (out[0].permute(1, 2, 0).float().cpu().numpy() * 0.5 + 0.5) * 255.0
    return out.clip(0, 255).astype(np.uint8)


if __name__ == "__main__":
    torch.cuda.init()
    torch.zeros(1, device=DEVICE)
    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()
    tok, txt, sched, unet, vae = build_difix()
    print(f"load: {time.time() - t0:.1f}s")

    rng = np.random.default_rng(0)
    dummy = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    t1 = time.time()
    out = difix_fix(dummy, tok, txt, sched, unet, vae, prompt="")
    torch.cuda.synchronize(0)
    print(f"forward: {time.time() - t1:.2f}s  out shape={out.shape} dtype={out.dtype} "
          f"range=({out.min()},{out.max()})")
    print(f"PEAK_VRAM_GB={torch.cuda.max_memory_allocated(0) / 1e9:.2f}")
    print("DIFIX_FORWARD_OK")


def _sanity_structured():
    import numpy as _np
    tok, txt, sched, unet, vae = build_difix()
    # smooth gradient with a blurry blob -> a real degraded-render-like image
    yy, xx = _np.mgrid[0:512, 0:512]
    img = _np.stack([
        (xx / 511 * 255),
        (yy / 511 * 255),
        ((xx + yy) / 1022 * 255),
    ], -1).astype(_np.uint8)
    out = difix_fix(img, tok, txt, sched, unet, vae, prompt="remove artifacts")
    d = _np.abs(out.astype(float) - img.astype(float))
    print(f"STRUCT out range=({out.min()},{out.max()}) mean_abs_delta={d.mean():.2f} "
          f"std_out={out.std():.1f}")
