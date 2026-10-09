"""CPU regression tests using small official iMF networks and a local VAE.

Run: python -m unittest discover -s tests -v
No pretrained downloads, ImageNet, or CUDA are required.
"""

import contextlib
import functools
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from diffusers import AutoencoderKL

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config
from models.imf_torch import imfDiT
from models.imf_torch.imf import iMeanFlow
from models.imf_torch.registry import model_defaults
from models.model import TokenizerFlowComposition
from scripts import generate_eval_images


class DecoderAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.context = contextlib.ExitStack()
        cls.addClassCleanup(cls.context.close)
        cls.root = Path(cls.context.enter_context(tempfile.TemporaryDirectory()))
        tiny_factory = functools.partial(
            imfDiT.imfDiT, hidden_size=32, depth=2, aux_head_depth=1,
            num_heads=4, patch_size=2,
        )
        cls.context.enter_context(patch.object(imfDiT, "imfDiT_B_2", tiny_factory))
        # Compile is a GPU evaluation optimization, independent of decoder loading.
        cls.context.enter_context(patch.object(torch, "compile", lambda fn, **kwargs: fn))
        torch.manual_seed(7)
        vae = AutoencoderKL(
            in_channels=3, out_channels=3, latent_channels=4,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(8,), layers_per_block=1, norm_num_groups=4,
            sample_size=32, mid_block_add_attention=False,
        )
        vae.save_pretrained(cls.root / "vae")
        cls.context.enter_context(patch.dict(os.environ, {"SD_VAE_PATH": str(cls.root / "vae")}))
        torch.save(iMeanFlow("imfDiT_B_2", eval=True).state_dict(), cls.root / "imf.pth")

    def args(self, use_cfg=False, **overrides):
        values = dict(
            model_type="iMF-B-2", pretrained_imf_pytorch=str(self.root / "imf.pth"),
            vae_type="mse", stage="train_decoder", latent_size=32,
            minimum_noise_level=0.25, maximum_noise_level=0.45,
            fixed_noise=True, normalized=False, adaptation_use_cfg=use_cfg,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def build(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return TokenizerFlowComposition(self.args(*args, **kwargs))

    def run_generation(self, argv):
        env = {key: value for key, value in os.environ.items()
               if key not in ("RANK", "WORLD_SIZE", "LOCAL_RANK")}
        with contextlib.ExitStack() as context:
            context.enter_context(patch.object(sys, "argv", ["generate_eval_images"] + argv))
            context.enter_context(patch.dict(os.environ, env, clear=True))
            context.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
            context.enter_context(contextlib.redirect_stdout(io.StringIO()))
            context.enter_context(contextlib.redirect_stderr(io.StringIO()))
            generate_eval_images.main()

    def test_cli_defaults_cfg_overrides_and_run_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = [
                "train", "--pretrained_imf_pytorch", str(self.root / "imf.pth"),
                "--minimum_noise_level", "0.25", "--maximum_noise_level", "0.45",
            ]
            for option in ("checkpoint_dir", "results_dir", "saver_dir", "yaml_dir"):
                base += ["--" + option, str(Path(tmp) / option)]
            for enabled in (False, True):
                flags = ["--adaptation_use_cfg", "--omega", "6.5", "--t_min", "0.2",
                         "--t_max", "0.8"] if enabled else []
                with patch.object(sys, "argv", base + flags):
                    args = config.parse_arg()
                self.assertEqual(args.adaptation_use_cfg, enabled)
                self.assertEqual(args.saver_name_pre.endswith("_CFG"), enabled)
                self.assertEqual(args.omega, 6.5 if enabled else model_defaults("iMF-B-2")["omega"])

    def test_cli_rejects_invalid_noise_cfg_and_checkpoint(self):
        for flags in (
            ["--minimum_noise_level", "0.5", "--maximum_noise_level", "0.25"],
            ["--t_min", "0.8", "--t_max", "0.2"],
            ["--omega", "0"], ["--eval_epochs", "0"],
            ["--pretrained_imf_pytorch", "legacy.pt"],
        ):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as tmp:
                argv = ["train", "--pretrained_imf_pytorch", "model.pth"]
                for option in ("checkpoint_dir", "results_dir", "saver_dir", "yaml_dir"):
                    argv += ["--" + option, str(Path(tmp) / option)]
                with patch.object(sys, "argv", argv + flags), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        config.parse_arg()
                self.assertEqual(error.exception.code, 2)

    def test_legacy_checkpoint_rejected_before_model_allocation(self):
        with patch("models.imf_torch.imf.iMeanFlow") as factory:
            with self.assertRaisesRegex(ValueError, "official .pth checkpoint"):
                self.build(pretrained_imf_pytorch=str(self.root / "legacy.pt"))
            factory.assert_not_called()
        with self.assertRaisesRegex(ValueError, "official .pth checkpoint"):
            self.run_generation([
                "--pretrained_imf_pytorch", "legacy.pt", "--modes", "gfid_cfg",
                "--output_dir", str(self.root / "invalid"),
            ])

    def test_renaming_converted_weights_cannot_bypass_strict_loading(self):
        renamed = self.root / "renamed.pth"
        torch.save({"state_dict": {"converted.weight": torch.zeros(1)}}, renamed)
        with self.assertRaises(RuntimeError):
            self.build(pretrained_imf_pytorch=str(renamed))

    def test_training_cfg_reaches_official_sampler(self):
        images, labels = torch.randn(2, 3, 32, 32), torch.tensor([1, 2])
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                model = self.build(enabled, omega=6.5, t_min=0.2, t_max=0.8)
                prior = model.diffusion_pytorch
                with patch.object(prior.model, "sample_one_step", wraps=prior.model.sample_one_step) as sample:
                    output = model(images, labels)
                    actual = tuple(float(value) for value in sample.call_args.args[4:7])
                    expected = (6.5, 0.2, 0.8) if enabled else (1.0, 0.0, 1.0)
                    for value, target in zip(actual, expected):
                        self.assertAlmostEqual(value, target, places=6)
                self.assertEqual(output.shape, images.shape)
                self.assertTrue(torch.isfinite(output).all())
                self.assertFalse(any(p.requires_grad for p in prior.parameters()))
                self.assertFalse(any(p.requires_grad for p in model.tokenizer.vae.encoder.parameters()))
                self.assertTrue(all(p.requires_grad for p in model.tokenizer.vae.decoder.parameters()))
                with patch.object(prior, "denoise_from_encoder_latent",
                                  wraps=prior.denoise_from_encoder_latent) as denoise:
                    model.collect_eval_info(images, labels)
                    self.assertEqual(denoise.call_args.kwargs["use_cfg"], enabled)

    def test_evaluation_cfg_overrides(self):
        model = self.build(omega=7.0, cfg_omega=9.0, cfg_t_min=0.3, cfg_t_max=0.7)
        prior = model.diffusion_pytorch
        self.assertEqual((prior.omega, prior.t_min, prior.t_max), (9.0, 0.3, 0.7))

    def test_official_gar_noise_and_normalization_contract(self):
        prior = self.build().diffusion_pytorch
        latent = torch.randn(2, 4, 32, 32, requires_grad=True)
        labels = torch.tensor([3, 4])

        def oracle(z, noise_level, labels=None, num_steps=1, use_cfg=False):
            return z * 0.75 + noise_level * (2.0 if use_cfg else 1.0)

        for normalized in (False, True):
            for fixed_noise in (False, True):
                for use_cfg in (False, True):
                    with self.subTest(normalized=normalized, fixed_noise=fixed_noise, use_cfg=use_cfg):
                        torch.manual_seed(123)
                        level = 0.45 if fixed_noise else 0.25 + torch.rand(1).item() * 0.2
                        noisy = (1 - level) * latent.detach() + level * torch.randn_like(latent)
                        if normalized:
                            mean = noisy.mean(dim=(1, 2, 3), keepdim=True)
                            std = noisy.std(dim=(1, 2, 3), keepdim=True)
                            noisy = (noisy - mean) / (std + 1e-8)
                        expected = oracle(noisy, level, use_cfg=use_cfg)
                        if normalized:
                            expected = expected * std + mean
                        torch.manual_seed(123)
                        with patch.object(prior, "denoise", side_effect=oracle) as denoise:
                            output = prior.denoise_from_encoder_latent(
                                latent, 0.45, normalized=normalized, labels=labels,
                                num_steps=2, use_cfg=use_cfg, fixed_noise=fixed_noise,
                                minimum_noise_level=0.25,
                            )
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                        self.assertFalse(output.requires_grad)
                        self.assertIs(denoise.call_args.kwargs["labels"], labels)
                        self.assertEqual(denoise.call_args.kwargs["num_steps"], 2)
                        self.assertEqual(denoise.call_args.kwargs["use_cfg"], use_cfg)

    def test_official_noise_bounds_and_zero_endpoint(self):
        prior = self.build().diffusion_pytorch
        latent, labels = torch.randn(2, 4, 32, 32), torch.tensor([1, 2])
        with patch.object(prior, "denoise", wraps=prior.denoise) as denoise:
            for _ in range(4):
                prior.denoise_from_encoder_latent(latent, 0.45, labels=labels, minimum_noise_level=0.25)
            levels = [call.args[1] for call in denoise.call_args_list]
            self.assertTrue(all(0.25 <= level <= 0.45 for level in levels))
            self.assertGreater(len(set(levels)), 1)
        torch.testing.assert_close(
            prior.denoise_from_encoder_latent(latent, 0.0, labels=labels, fixed_noise=True), latent,
        )
        for minimum, maximum in ((0.5, 0.25), (-0.1, 0.4), (0.0, 1.1)):
            with self.subTest(minimum=minimum, maximum=maximum), self.assertRaises(ValueError):
                prior.denoise_from_encoder_latent(latent, maximum, minimum_noise_level=minimum)
        with self.assertRaises(ValueError):
            prior.denoise_from_encoder_latent(latent, 0.4, num_steps=0)

    def test_decoder_update_checkpoint_resume_and_official_eval_sync(self):
        images, labels = torch.randn(2, 3, 32, 32), torch.tensor([1, 2])
        model = self.build(True)
        optimizer = torch.optim.AdamW(model.tokenizer.vae.decoder.parameters(), lr=1e-3)
        before = model.tokenizer.vae.decoder.conv_out.weight.detach().clone()
        output = model(images, labels)
        torch.nn.functional.mse_loss(output, images).backward()
        self.assertTrue(any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.tokenizer.vae.decoder.parameters()
        ))
        self.assertTrue(all(p.grad is None for p in model.diffusion_pytorch.parameters()))
        self.assertTrue(all(p.grad is None for p in model.tokenizer.vae.encoder.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.tokenizer.vae.decoder.conv_out.weight))
        checkpoint_path = self.root / "resume.pth.tar"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 1},
                   checkpoint_path)
        restored = self.build(True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        restored.load_state_dict(checkpoint["model"], strict=True)
        resumed_optimizer = torch.optim.AdamW(restored.tokenizer.vae.decoder.parameters(), lr=1e-3)
        resumed_optimizer.load_state_dict(checkpoint["optimizer"])
        self.assertEqual(len(resumed_optimizer.state), len(optimizer.state))
        restored.eval_mode()
        latent = torch.randn(2, 4, 32, 32)
        keys_before = set(restored.state_dict())
        with torch.no_grad():
            expected = restored.tokenizer.decode(latent).clamp(-1, 1)
            self.assertFalse(torch.allclose(restored.decode_latent(latent), expected, atol=1e-5))
            restored.sync_eval_decoder()
            torch.testing.assert_close(restored.decode_latent(latent), expected, rtol=1e-4, atol=1e-5)
        self.assertEqual(set(restored.state_dict()), keys_before)
        self.assertTrue(all(not p.requires_grad for p in restored.vae_wrapper.parameters()))

    def test_generation_entrypoint_loads_adapted_decoder(self):
        # A constant decoder output reveals stale evaluation weights in the NPZ.
        model = self.build()
        with torch.no_grad():
            model.tokenizer.vae.decoder.conv_out.weight.zero_()
            model.tokenizer.vae.decoder.conv_out.bias.copy_(torch.tensor([0.25, -0.5, 0.75]))
        official_state = model.state_dict()
        # Previous converted-backend checkpoints used the same tokenizer keys.
        legacy_state = {key: value for key, value in official_state.items() if key.startswith("tokenizer.")}
        legacy_state["diffusion_pytorch.model.u_final_layer.linear.weight"] = torch.zeros(1)
        for layout, state in (("official", official_state), ("legacy", legacy_state)):
            with self.subTest(layout=layout):
                checkpoint = self.root / f"adapted_{layout}.pth.tar"
                torch.save({"model": state, "args": vars(model.args)}, checkpoint)
                output_dir = self.root / f"generated_{layout}"
                self.run_generation([
                    "--pretrained_imf_pytorch", str(self.root / "imf.pth"),
                    "--pretrained_decoder", str(checkpoint), "--output_dir", str(output_dir),
                    "--num_samples", "2", "--batch_size", "2", "--modes", "gfid_cfg,gfid_nocfg",
                    "--cfg_omega", "6.5", "--cfg_t_min", "0.2", "--cfg_t_max", "0.8",
                ])
                for tag in ("CFG", "noCFG"):
                    with np.load(output_dir / f"Generated_{tag}_iMF-B-2.npz") as archive:
                        images = archive["arr_0"]
                    self.assertEqual(images.shape, (2, 32, 32, 3))
                    self.assertEqual(images.dtype, np.uint8)
                    expected = np.broadcast_to(np.array([159, 64, 223], dtype=np.uint8), images.shape)
                    np.testing.assert_array_equal(images, expected)

    def test_generation_entrypoint_rejects_checkpoint_without_decoder(self):
        checkpoint = self.root / "encoder_only.pth.tar"
        torch.save({"model": {"tokenizer.mean": torch.zeros(1, 4, 1, 1)}}, checkpoint)
        with self.assertRaisesRegex(ValueError, "no matching tokenizer.vae.decoder"):
            self.run_generation([
                "--pretrained_imf_pytorch", str(self.root / "imf.pth"),
                "--pretrained_decoder", str(checkpoint), "--modes", "gfid_cfg",
                "--output_dir", str(self.root / "invalid"),
            ])

    def test_debug_mode_without_generator(self):
        model = self.build(debug_mode=True, pretrained_imf_pytorch="")
        images, labels = torch.randn(2, 3, 32, 32), torch.tensor([1, 2])
        self.assertIsNone(model.diffusion_pytorch)
        self.assertIsNone(model.vae_wrapper)
        expected = model.tokenizer.decode(model.tokenizer.encode(images)).clamp(-1, 1)
        torch.testing.assert_close(model(images, labels), expected)


if __name__ == "__main__":
    unittest.main()
