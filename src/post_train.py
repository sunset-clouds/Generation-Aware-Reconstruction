"""Compatibility entrypoint for src/train_decoder_adaptation.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train_decoder_adaptation.py")), run_name="__main__")
else:
    from train_decoder_adaptation import is_main_process, main_worker
