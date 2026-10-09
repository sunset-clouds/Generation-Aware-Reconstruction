"""CPU regression tests using small real iMF networks and a local Diffusers VAE.

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
from models.imf_torch.converted_arch import MODEL_CONFIGS, MiT_PyTorch
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
        tiny_config = dict(MODEL_CONFIGS["iMF-B-2"])
        tiny_config.update(hidden_size=32, num_shared_blocks=1, num_head_blocks=1, num_heads=4)
        cls.context.enter_context(patch.dict(MODEL_CONFIGS, {"iMF-B-2": tiny_config}))
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
        converted = MiT_PyTorch(input_size=32, **tiny_config)
        torch.save({"state_dict": converted.state_dict()}, cls.root / "imf.pt")

    def args(self, suffix=".pth", use_cfg=False, **overrides):
        values = dict(
            model_type="iMF-B-2", pretrained_imf_pytorch=str(self.root / ("imf" + suffix)),
            vae_type="mse", stage="train_decoder", latent_size=32,
            minimum_noise_level=0.25, maximum_noise_level=0.45,
            fixed_noise=True, normalized=False, adaptation_use_cfg=use_cfg,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def build(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return TokenizerFlowComposition(self.args(*args, **kwargs))

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

    def test_cli_rejects_invalid_noise_and_cfg_ranges(self):
        for flags in (
            ["--minimum_noise_level", "0.5", "--maximum_noise_level", "0.25"],
            ["--t_min", "0.8", "--t_max", "0.2"],
            ["--omega", "0"], ["--eval_epochs", "0"],
        ):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as tmp:
                argv = ["train", "--pretrained_imf_pytorch", "model.pth"]
                for option in ("checkpoint_dir", "results_dir", "saver_dir", "yaml_dir"):
                    argv += ["--" + option, str(Path(tmp) / option)]
                with patch.object(sys, "argv", argv + flags), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        config.parse_arg()
                self.assertEqual(error.exception.code, 2)

    def test_training_cfg_reaches_both_backends(self):
        images, labels = torch.randn(2, 3, 32, 32), torch.tensor([1, 2])
        for suffix in (".pt", ".pth"):
            for enabled in (False, True):
                with self.subTest(suffix=suffix, enabled=enabled):
                    model = self.build(suffix, enabled, omega=6.5, t_min=0.2, t_max=0.8)
                    prior = model.diffusion_pytorch
                    with patch.object(prior.model, "sample_one_step",
                                      wraps=prior.model.sample_one_step) as sample:
                        output = model(images, labels)
                        call = sample.call_args.args
                        actual = tuple(float(value) for value in call[4:7])
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

    def test_evaluation_cfg_overrides_reach_both_backends(self):
        for suffix in (".pt", ".pth"):
            with self.subTest(suffix=suffix):
                model = self.build(suffix, omega=7.0, cfg_omega=9.0, cfg_t_min=0.3, cfg_t_max=0.7)
                prior = model.diffusion_pytorch
                self.assertEqual((prior.omega, prior.t_min, prior.t_max), (9.0, 0.3, 0.7))

    def test_official_gar_matches_validated_converted_helper(self):
        # Use the same denoising oracle for both wrappers to isolate the GAR
        # construction contract (NCHW, CFG, noise sampling, normalization).
        official = self.build(".pth").diffusion_pytorch
        converted = self.build(".pt").diffusion_pytorch
        latent = torch.randn(2, 4, 32, 32, requires_grad=True)
        labels = torch.tensor([3, 4])

        def oracle(z, noise_level, labels=None, num_steps=1, use_cfg=False):
            return z * 0.75 + noise_level * (2.0 if use_cfg else 1.0)

        for normalized in (False, True):
            for fixed_noise in (False, True):
                for use_cfg in (False, True):
                    with self.subTest(normalized=normalized, fixed_noise=fixed_noise, use_cfg=use_cfg):
                        outputs = []
                        for prior in (official, converted):
                            torch.manual_seed(123)
                            with patch.object(prior, "denoise", side_effect=oracle):
                                outputs.append(prior.denoise_from_encoder_latent(
                                    latent, 0.45, normalized=normalized, labels=labels,
                                    use_cfg=use_cfg, fixed_noise=fixed_noise,
                                    minimum_noise_level=0.25,
                                ))
                        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                        self.assertFalse(outputs[0].requires_grad)

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
        for suffix in (".pt", ".pth"):
            with self.subTest(suffix=suffix):
                model = self.build(suffix, True)
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
                checkpoint_path = self.root / ("resume" + suffix)
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 1},
                           checkpoint_path)
                restored = self.build(suffix, True)
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
                    if suffix == ".pth":
                        self.assertFalse(torch.allclose(restored.decode_latent(latent), expected, atol=1e-5))
                    restored.sync_eval_decoder()
                    torch.testing.assert_close(restored.decode_latent(latent), expected, rtol=1e-4, atol=1e-5)
                self.assertEqual(set(restored.state_dict()), keys_before)
                if suffix == ".pth":
                    self.assertTrue(all(not p.requires_grad for p in restored.vae_wrapper.parameters()))

    def test_generation_entrypoint_loads_adapted_decoder_for_both_backends(self):
        # A constant decoder output makes stale evaluation weights observable in
        # the final NPZ, including the real CLI checkpoint-filtering path.
        env = {key: value for key, value in os.environ.items()
               if key not in ("RANK", "WORLD_SIZE", "LOCAL_RANK")}
        for suffix in (".pt", ".pth"):
            with self.subTest(suffix=suffix):
                model = self.build(suffix)
                with torch.no_grad():
                    model.tokenizer.vae.decoder.conv_out.weight.zero_()
                    model.tokenizer.vae.decoder.conv_out.bias.copy_(torch.tensor([0.25, -0.5, 0.75]))
                checkpoint = self.root / ("adapted" + suffix)
                torch.save({"model": model.state_dict(), "args": vars(model.args)}, checkpoint)
                output_dir = self.root / ("generated" + suffix)
                argv = [
                    "generate_eval_images", "--pretrained_imf_pytorch",
                    str(self.root / ("imf" + suffix)), "--pretrained_decoder", str(checkpoint),
                    "--output_dir", str(output_dir), "--num_samples", "2", "--batch_size", "2",
                    "--modes", "gfid_cfg,gfid_nocfg", "--cfg_omega", "6.5",
                    "--cfg_t_min", "0.2", "--cfg_t_max", "0.8",
                ]
                with patch.object(sys, "argv", argv), patch.dict(os.environ, env, clear=True), \
                        patch.object(torch.cuda, "is_available", return_value=False), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    generate_eval_images.main()
                for tag in ("CFG", "noCFG"):
                    with np.load(output_dir / f"Generated_{tag}_iMF-B-2.npz") as archive:
                        images = archive["arr_0"]
                    self.assertEqual(images.shape, (2, 32, 32, 3))
                    self.assertEqual(images.dtype, np.uint8)
                    expected = np.broadcast_to(np.array([159, 64, 223], dtype=np.uint8), images.shape)
                    np.testing.assert_array_equal(images, expected)

    def test_generation_entrypoint_rejects_checkpoint_without_decoder(self):
        env = {key: value for key, value in os.environ.items()
               if key not in ("RANK", "WORLD_SIZE", "LOCAL_RANK")}
        checkpoint = self.root / "encoder_only.pth.tar"
        torch.save({"model": {"tokenizer.mean": torch.zeros(1, 4, 1, 1)}}, checkpoint)
        for suffix in (".pt", ".pth"):
            with self.subTest(suffix=suffix):
                argv = [
                    "generate_eval_images", "--pretrained_imf_pytorch",
                    str(self.root / ("imf" + suffix)), "--pretrained_decoder", str(checkpoint),
                    "--output_dir", str(self.root / "invalid"), "--modes", "gfid_cfg",
                ]
                with patch.object(sys, "argv", argv), patch.dict(os.environ, env, clear=True), \
                        patch.object(torch.cuda, "is_available", return_value=False), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "no matching tokenizer.vae.decoder"):
                        generate_eval_images.main()


if __name__ == "__main__":
    unittest.main()
