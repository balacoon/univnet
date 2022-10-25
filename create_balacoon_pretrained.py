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


def parse_args():
    ap = argparse.ArgumentParser(
        description="Traces univnet for balacoon/pretrained and packs addon for balacoon_backend"
    )
    ap.add_argument(
        "--ckpt",
        default="univ_c32_0288.pt",
        required=True,
        help="Path to pretrained Univnet model (https://github.com/clementruhm/univnet#pre-trained-model)",
    )
    ap.add_argument(
        "--out-dir",
        default="./exported",
        help="Directory to put exported files to",
    )
    ap.add_argument("--use-gpu", action="store_true", help="Whether to trace on GPU")
    args = ap.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if args.use_gpu else "cpu")

    # load checkpoint
    loaded = torch.load(args.ckpt)
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
    ).to(device)
    stft.eval()
    logging.info("Created mel extractor")

    # run created model on example
    melspec_real = None
    example_wav = "arctic_a0001.wav"
    logging.info("Extracting melspec from real audio")
    audio_real, sample_rate = soundfile.read(example_wav)
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

    # export model to a file
    stft_path = os.path.join(args.out_dir, "univnet_melspec_extractor.jit")
    stft_traced = torch.jit.trace(stft, audio_real)
    stft_traced.save(stft_path)
    logging.info("Saved traced melspec extractor to {}".format(stft_path))

    # create waveform generator
    generator = Generator(config, squeeze_output=True, output_short=True, half_precision=False)
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
    generator = generator.to(device)
    generator.eval(inference=True)
    logging.info("Created audio generator")

    # run created generator on example data
    logging.info("Generating audio from melspec extractor previously")
    noise_real = torch.randn_like(melspec_real)[:, : generator.noise_dim]
    samples_real = generator(melspec_real, noise_real).cpu().detach().numpy()
    assert len(samples_real.shape) == 2, "expected output of generator is 2d"
    assert (
        samples_real.shape[0] == 1
    ), "expects to return single sequence as output of generator"
    samples_real = samples_real[0]
    resynt_path = os.path.join(args.out_dir, "resynt.wav")
    logging.info("Write re-synthesized audio to {}".format(resynt_path))
    soundfile.write(resynt_path, samples_real, config.audio.sampling_rate)

    # example of input to waveform generator: melspec and noise
    generator_path = os.path.join(args.out_dir, "univnet_vocoder.jit")
    generator_traced = torch.jit.trace(generator, [melspec_real, noise_real])
    generator_traced.save(generator_path)
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
    with open(generator_path, "rb") as fp:
        addon["synthesizer"] = fp.read()
    addon_path = os.path.join(args.out_dir, "univnet_vocoder.addon")
    with open(addon_path, "wb") as fp:
        msgpack.dump([addon], fp)
    logging.info("Saved addon for balacoon_backend to {}".format(addon_path))


if __name__ == "__main__":
    main()
