"""Generative :class:`ViewFiller` — the OFFICIAL reference-conditioned Difix3D+ artifact-fixer.

This turns a coverage-gap pose's *degraded* splat render into a *clean* target image with the
single-step Difix prior, conditioned on the closest clean **reference** training view, then the
orchestrator distils that target back into the gaussians so the under-observed region gains
plausible, multi-view-consistent content (the Difix3D+ recipe at 24 GB).

What it does, per gap view:

1. Run the vendored :class:`DifixPipeline` (``nvidia/difix_ref``) on the degraded render,
   conditioned on a reference image (``ref_image``). The pipeline batches ``[image, ref_image]``
   and the reference-mixing self-attention in the custom UNet lets the degraded view borrow
   appearance from the clean reference — this cross-view conditioning is the core of Difix3D+ and
   is exactly what the prior no-reference path threw away. The published recipe is a SINGLE step
   at ``τ=199`` (the default); ``num_inference_steps>1`` walks a descending τ-ladder instead, and
   ``sdedit=True`` additionally starts from the ArtiFixer opacity-mix latent init
   ``z_mix = O_z*z_deg + (1-O_z)*eps`` (real generative fill in fully-empty latent cells).
2. **Hard recomposite** ``target = M*generated + (1-M)*render`` with ``M = coverage_mask(alpha)``
   so trusted (high-alpha) pixels are taken verbatim from the real render and never disturbed;
   only the holes carry the generated content into the distiller.

The pipeline is loaded lazily / staged (mirroring ``DA3DepthEstimator``): nothing touches the GPU
until the first :meth:`fill`. :meth:`free` unloads the model and releases VRAM so the distiller
can own the card afterwards. Probed on an RTX 4090: 5.2 GB to load, 8.4 GB peak per forward.

If constructed with ``filler_mode="geometric-only"`` this class is import-safe but raises a clear
:class:`RuntimeError` on :meth:`fill`, so the geometric fallback is the path. Diffusion deps
(``diffusers``/``transformers``/``peft``) are imported lazily inside the loader; a missing dep
surfaces as a clear :class:`ImportError` with the install line.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gaussian_robot.enhance.mask import coverage_mask, downscale_to_latent
from gaussian_robot.enhance.protocols import SupervisionView

if TYPE_CHECKING:
    from gaussian_robot.render.base import RenderResult

_INSTALL_HINT = (
    "DiffusionFiller needs diffusers/transformers/peft/accelerate/safetensors. Install with: "
    "uv pip install --python <venv>/bin/python diffusers accelerate peft safetensors"
)

# Published Difix recipe constants (see Difix3D+ paper / nvidia/difix_ref model card).
_DEFAULT_REPO = "nvidia/difix_ref"
_DEFAULT_TIMESTEP = 199
_DEFAULT_PROMPT = "remove degradation"


def load_difix_pipeline(
    repo: str = _DEFAULT_REPO,
    *,
    device: str = "cuda:0",
    dtype: Any = None,
) -> Any:
    """Assemble the reference-conditioned Difix pipeline from the vendored classes + Hub weights.

    Each component is loaded into the LOCAL vendored class (``_difix/``), not the Hub's remote-code
    copy, so the only network access is the one-time weight download. The safety checker is disabled
    (the published ``model_index.json`` ships it as ``null``). Returns a ``DifixPipeline``.
    """
    from diffusers import DDPMScheduler  # noqa: PLC0415
    from transformers import CLIPTextModel, CLIPTokenizer  # noqa: PLC0415

    from gaussian_robot.enhance.fillers._difix.autoencoder_kl import AutoencoderKL  # noqa: PLC0415
    from gaussian_robot.enhance.fillers._difix.pipeline_difix import DifixPipeline  # noqa: PLC0415
    from gaussian_robot.enhance.fillers._difix.unet_2d_condition import (  # noqa: PLC0415
        UNet2DConditionModel,
    )

    tok = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
    txt = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder")
    sched = DDPMScheduler.from_pretrained(repo, subfolder="scheduler")  # type: ignore[no-untyped-call]
    unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(repo, subfolder="vae")

    pipe = DifixPipeline(
        vae=vae,
        text_encoder=txt,
        tokenizer=tok,
        unet=unet,
        scheduler=sched,
        safety_checker=None,
        feature_extractor=None,
        image_encoder=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    if dtype is not None:
        pipe = pipe.to(dtype)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _to_u8(img: np.ndarray) -> np.ndarray:
    """``(H, W, 3)`` array (uint8 or float [0,1]) -> contiguous uint8."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(a)


class DiffusionFiller:
    """Reference-conditioned single-step Difix :class:`ViewFiller` with hard recomposite.

    Parameters
    ----------
    filler_mode:
        ``"difix"`` loads the generative model; ``"geometric-only"`` makes this class import-safe
        but :meth:`fill` raises (so a separate :class:`GeometricFiller` is the path).
    model:
        HF repo of the diffusers-format Difix weights (``nvidia/difix_ref`` = reference-conditioned).
    device:
        CUDA device for the model.
    dtype:
        Torch dtype; ``"float32"`` matches NVIDIA's reference tests (≈8 GB). ``"float16"`` halves it.
    timestep:
        Difix denoise timestep τ (published default 199) — the single fixed step, or the TOP of
        the descending ladder when ``num_inference_steps > 1``.
    num_inference_steps:
        ``1`` (default) is the published single-step recipe. ``N > 1`` denoises through a
        descending τ-ladder ``τ, τ·(N-1)/N, …, τ/N`` — changes output character; gate on held-out
        PSNR (docs/research/artifixer-closeness-24gb.md Step 2).
    sdedit:
        Start from the ArtiFixer opacity-mix latent init (``z_mix = O_z*z_deg + (1-O_z)*eps``,
        noised to the top of the ladder) so fully-empty latent cells are truly generated rather
        than decoded from an empty render. Requires ``num_inference_steps >= 2``: on a single
        fixed-τ step the noised init cannot be denoised and only degrades output.
    prompt:
        Conditioning text (published default ``"remove degradation"``).
    tau_lo, feather:
        Coverage-mask thresholds (forwarded to :func:`coverage_mask`) for the recomposite; also
        the source of the ``O_z`` pooling when ``sdedit=True``.
    """

    def __init__(
        self,
        *,
        filler_mode: str = "difix",
        model: str = _DEFAULT_REPO,
        device: str = "cuda:0",
        dtype: str = "float32",
        timestep: int = _DEFAULT_TIMESTEP,
        num_inference_steps: int = 1,
        sdedit: bool = False,
        prompt: str = _DEFAULT_PROMPT,
        tau_lo: float = 0.5,
        feather: float = 0.15,
        disagreement: bool = True,
        dis_lo: float = 0.03,
        dis_hi: float = 0.12,
    ) -> None:
        if filler_mode not in ("difix", "geometric-only"):
            raise ValueError(
                f"filler_mode must be 'difix' or 'geometric-only', got {filler_mode!r}"
            )
        if num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
        if sdedit and num_inference_steps < 2:
            raise ValueError(
                "sdedit=True requires num_inference_steps >= 2: a single fixed-τ step cannot "
                "denoise the noised opacity-mix init — it would only inject noise "
                "(docs/research/artifixer-closeness-24gb.md Step 3)."
            )
        self._mode = filler_mode
        self._repo = model
        self._device = device
        self._dtype_name = dtype
        self._timestep = int(timestep)
        self._num_steps = int(num_inference_steps)
        self._sdedit = sdedit
        self._prompt = prompt
        self._tau_lo = tau_lo
        self._feather = feather
        self._disagreement = disagreement
        self._dis_lo = dis_lo
        self._dis_hi = dis_hi
        self._pipe: Any = None

    # ---- lifecycle ----------------------------------------------------------------------

    def _torch_dtype(self) -> Any:
        import torch  # noqa: PLC0415

        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
            self._dtype_name
        ]

    def load(self) -> None:
        """Stage the Difix pipeline onto the GPU (idempotent). Called lazily by :meth:`fill`."""
        if self._mode == "geometric-only":
            raise RuntimeError(
                "DiffusionFiller(filler_mode='geometric-only') cannot load weights; "
                "use GeometricFiller for the no-weights path."
            )
        if self._pipe is not None:
            return
        try:
            self._pipe = load_difix_pipeline(
                self._repo, device=self._device, dtype=self._torch_dtype()
            )
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise ImportError(_INSTALL_HINT) from exc

    def free(self) -> None:
        """Unload the model and release VRAM (so the distiller can own the card)."""
        self._pipe = None
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    # alias to match the DA3-style "unload" verb requested by the spec.
    unload = free

    # ---- the generative step ------------------------------------------------------------

    def _denoise_timesteps(self) -> list[int]:
        """Descending τ-ladder for the scheduler: ``[τ]`` single-step, else ``τ·(N-i)/N``.

        The published Difix recipe is one step at τ=199; the multi-step ladder walks from τ down
        toward 0 in equal fractions (the final scheduler step lands on the clean sample). Entries
        are deduped/strictly descending so tiny τ with large N stays a valid schedule.
        """
        n = self._num_steps
        raw = [max(1, round(self._timestep * (n - i) / n)) for i in range(n)]
        ladder = [t for i, t in enumerate(raw) if i == 0 or t < raw[i - 1]]
        return ladder

    def _difix(
        self,
        render_u8: np.ndarray,
        ref_u8: np.ndarray | None,
        alpha: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reference-conditioned Difix forward. Returns ``(H, W, 3)`` float [0,1].

        ``alpha`` (H, W) is only consumed when ``sdedit=True``: it is max-pooled to latent
        resolution (``O_z``) and drives the opacity-mix latent init inside the vendored pipeline.
        """
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        h, w = render_u8.shape[:2]
        nh, nw = h - h % 8, w - w % 8  # VAE is f=8; operate on a multiple of 8
        image = Image.fromarray(render_u8[:nh, :nw])
        ref = Image.fromarray(ref_u8[:nh, :nw]) if ref_u8 is not None else None

        init_mask = None
        if self._sdedit:
            a = (
                torch.as_tensor(alpha[:nh, :nw], dtype=torch.float32)
                if alpha is not None
                else torch.ones((nh, nw), dtype=torch.float32)
            )
            init_mask = downscale_to_latent(a, (nh // 8, nw // 8))

        timesteps = self._denoise_timesteps()
        out = self._pipe(
            self._prompt,
            image=image,
            ref_image=ref,
            num_inference_steps=len(timesteps),
            timesteps=timesteps,
            init_mask=init_mask,
            guidance_scale=0.0,
            output_type="np",
        ).images[0]
        gen = np.asarray(out, dtype=np.float32)

        if (nh, nw) != (h, w):  # restore full frame: pad cropped border with the raw render
            full = render_u8.astype(np.float32) / 255.0
            full[:nh, :nw] = gen
            return full.clip(0.0, 1.0)
        return gen.clip(0.0, 1.0)

    def fill(self, degraded: RenderResult, references: Sequence[RenderResult]) -> SupervisionView:
        """Difix-clean the degraded render (conditioned on the first reference) and recomposite.

        ``target_rgb = M*generated + (1-M)*render``. The first entry of ``references`` (the closest
        clean training view) is used as ``ref_image``; an empty ``references`` falls back to the
        no-reference forward.

        The mask ``M`` combines two signals (element-wise max):

        - ``coverage_mask(alpha)`` — transparent holes (where the splat is under-reconstructed).
        - **disagreement** (when ``disagreement=True``, default) — a smoothstep on the per-pixel
          ``|generated - render|``. Difix is an artifact-fixer: its main job is SHARPENING blurry
          but *opaque* regions (alpha≈1), which the alpha-only mask scored ~0 and therefore threw
          away — the recomposite was discarding ~all of Difix's work on well-covered interiors.
          The disagreement term is self-targeting: it trusts Difix exactly where it changed the
          image (the blurry gap), while leaving already-sharp regions (tiny diff) protected. The
          interleaved real anchors + the held-out PSNR gate guard against global drift.
        """
        if self._mode == "geometric-only":
            raise RuntimeError(
                "DiffusionFiller(filler_mode='geometric-only').fill() is disabled; "
                "use GeometricFiller for the no-weights path."
            )
        import torch  # noqa: PLC0415

        self.load()

        render_u8 = _to_u8(degraded.rgb)
        h, w = render_u8.shape[:2]
        alpha = (
            np.asarray(degraded.alpha, dtype=np.float32)
            if degraded.alpha is not None
            else np.ones((h, w), dtype=np.float32)
        )
        ref_u8 = _to_u8(references[0].rgb) if references else None

        with torch.no_grad():
            generated = self._difix(render_u8, ref_u8, alpha)

        mask = (
            coverage_mask(
                torch.as_tensor(alpha, dtype=torch.float32),
                tau_lo=self._tau_lo,
                feather=self._feather,
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        render = render_u8.astype(np.float32) / 255.0
        if self._disagreement:
            # Per-pixel mean-abs channel change Difix made; smoothstep it into [0,1].
            diff = np.abs(generated - render).mean(axis=-1)
            span = max(self._dis_hi - self._dis_lo, 1e-6)
            t = np.clip((diff - self._dis_lo) / span, 0.0, 1.0)
            dis = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
            mask = np.maximum(mask, dis)
        m3 = mask[..., None]
        target = (m3 * generated + (1.0 - m3) * render).clip(0.0, 1.0).astype(np.float32)
        return SupervisionView(camera=degraded.camera, target_rgb=target, mask=mask)
