"""
Copyright 2022 Balacoon

Script to export pretrained univnet as an addon,
compatible with balacoon_backend
"""
import argparse
import logging
import math
import os
from typing import Tuple, Union

import matplotlib.pylab as plt
import msgpack
import soundfile
import torch
from omegaconf import OmegaConf

from model.generator import Generator
from utils.stft import TacotronSTFT

BACKENDS = ["libtorch"]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Creates vocoder addon compatible with balacoon_backend"
    )
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Path to pretrained Univnet model (https://github.com/clementruhm/univnet#pre-trained-model)",
    )
    ap.add_argument("--out", required=True, help="Path to put created addon to")
    ap.add_argument(
        "--backend",
        default=BACKENDS[0],
        choices=BACKENDS,
        help="What is the backend to export to",
    )
    ap.add_argument(
        "--work-dir",
        default="./work_dir",
        help="Directory to put intermediate files to",
    )
    ap.add_argument(
        "--omit-generator",
        action="store_true",
        help="If provided, generator is not stored to addon",
    )
    ap.add_argument(
        "--wav", help="If specified, runs test analysis synthesis with traced model"
    )
    args = ap.parse_args()
    return args


def export(
    nn_module: torch.nn.Module,
    inputs: Union[Tuple[torch.Tensor], torch.Tensor],
    path: str,
    backend: str,
):
    """
    Given model, converts it to the graph and stores it to a file.
    Such model can be loaded for inference, without the need for any python
    code describing it.

    Parameters
    ----------
    nn_module: torch.nn.Module
        model to export
    inputs: Union[Tuple[torch.Tensor]
        example(s) of input for the model
    path: str
        location to store exported model to
    backend: str
        name of the backend to use for export
    """
    if backend == BACKENDS[0]:  # libtorch
        """
        run tracing and store the trace to a file
        """
        script = torch.jit.trace(nn_module, inputs)
        script.save(path)
    else:
        """
        # for the future, storing example how model should be exported for onnx backend
        elif backend == "onnx":
            torch.onnx.export(nn_module, inputs, path,
                  export_params=True, opset_version=17, do_constant_folding=True,
                  input_names=input_names, output_names=output_names,
                  dynamic_axes={input_names[0]: {0: "batch_size"}, output_names[0]: {0: "batch_size"}})
        """
        raise RuntimeError("Unsupported backend")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.work_dir, exist_ok=True)
    device = torch.device("cpu")

    # load checkpoint
    loaded = torch.load(args.checkpoint)
    config = OmegaConf.create(loaded["hp_str"])

    # create melspectrogram extractor with parameters
    # correspondent to trained model
    stft = TacotronSTFT(
        config.audio.filter_length,
        config.audio.hop_length,
        config.audio.win_length,
        config.audio.n_mel_channels,
        config.audio.sampling_rate,
        config.audio.mel_fmin,
        config.audio.mel_fmax,
        center=False,
        device=device,
    )
    stft.eval()
    logging.info("Created mel extractor")

    # run created model on example if one is provided
    melspec_real = None
    if args.wav:
        logging.info("Extracting melspec from real audio")
        audio_real, sample_rate = soundfile.read(args.wav)
        assert (
            sample_rate == config.audio.sampling_rate
        ), "Provided audio file should have {} sample rate".format(
            config.audio.sample_rate
        )
        audio_real = torch.tensor(
            audio_real, device=device, dtype=torch.float
        ).unsqueeze(
            0
        )  # (batch x samples)
        melspec_real = stft(audio_real)
        melspec_real_npy = melspec_real.cpu().detach().numpy()
        assert (
            len(melspec_real_npy.shape) == 3
        ), "expected output of melspec extractor is 3d"
        assert (
            melspec_real_npy.shape[0] == 1
        ), "melspec extractor should produce single output sequence"
        melspec_real_npy = melspec_real_npy[0]
        # plot the extracted melspectrogram for debug purposes
        plt.imshow(melspec_real_npy, aspect="auto")
        plt.show()

    # create dummy audio tensor, needed to trace the model
    audio_example = torch.randn(1, 24000, device=device, dtype=torch.float) - 0.5
    # export model to a file
    stft_path = os.path.join(args.work_dir, "stft.bin")
    export(stft, audio_example, stft_path, backend=args.backend)
    logging.info("Saved traced melspec extractor to {}".format(stft_path))

    # create waveform generator
    generator = Generator(config, squeeze_output=True)
    # rename parameters to have stand-alone generator
    saved_state_dict = loaded["model_g"]
    new_state_dict = {}
    # https://github.com/clementruhm/univnet/blob/master/inference.py#L23
    for k, v in saved_state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict["module." + k]
        except:
            new_state_dict[k] = v
    generator.load_state_dict(new_state_dict)
    generator.eval(inference=True)
    logging.info("Created audio generator")

    # run created generator on example data if provided
    if melspec_real is not None:
        logging.info("Generating audio from melspec extractor previously")
        noise_real = torch.randn_like(melspec_real)[:, : generator.noise_dim]
        samples_real = generator(melspec_real, noise_real).cpu().detach().numpy()
        assert len(samples_real.shape) == 2, "expected output of generator is 2d"
        assert (
            samples_real.shape[0] == 1
        ), "expects to return single sequence as output of generator"
        samples_real = samples_real[0]
        resynt_path = os.path.join(args.work_dir, "resynt.wav")
        logging.info("Write re-synthesized audio to {}".format(resynt_path))
        soundfile.write(resynt_path, samples_real, config.audio.sampling_rate)

    # example of input to waveform generator: melspec and noise
    melspec_example = torch.randn(
        1, config.audio.n_mel_channels, 100, device=device, dtype=torch.float
    )
    noise_example = torch.randn_like(melspec_example)[:, : generator.noise_dim]
    generator_path = os.path.join(args.work_dir, "generator.bin")
    export(
        generator,
        (melspec_example, noise_example),
        generator_path,
        backend=args.backend,
    )
    logging.info("Saved traced audio generator to {}".format(generator_path))

    # finally put all the exported models into the addon
    # TODO use balacoon_backend for addon field names
    addon = {
        "id": "vocoder",
        "sampling_rate": config.audio.sampling_rate,
        "rate_ratio": math.prod(config.gen.strides),
        "needs_noise": True,
        "noise_dimension": generator.noise_dim,
    }
    with open(stft_path, "rb") as fp:
        addon["analyzer"] = fp.read()
    if not args.omit_generator:
        with open(generator_path, "rb") as fp:
            addon["synthesizer"] = fp.read()
    with open(args.out, "wb") as fp:
        msgpack.dump([addon], fp)
    logging.info("Saved addon for balacoon_backend to {}".format(args.out))


if __name__ == "__main__":
    main()
