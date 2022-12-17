# MIT License
#
# Copyright (c) 2020 Jungil Kong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from nnAudio.features.stft import STFT


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=None, center=False, device='cuda', full_precision=False, skip_casts=False):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.n_fft = filter_length
        self.hop_size = hop_length
        self.win_size = win_length
        self.fmin = mel_fmin
        self.fmax = mel_fmax
        self.center = center
        self.full_precision = full_precision
        self.skip_casts = skip_casts
        self.pad_mode = "reflect" if self.full_precision else "zeros"
        # uses hann window by default
        self.stft = STFT(
            n_fft=self.n_fft,
            win_length=self.win_size,
            hop_length=self.hop_size,
            center=self.center,
            output_format="Magnitude",
            fmax=self.fmax,
            sr=self.sampling_rate,
        )

        mel = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)

        mel_basis = torch.from_numpy(mel).float().to(device)
        mel_basis = mel_basis if self.full_precision else mel_basis.half()
        hann_window = torch.hann_window(win_length).to(device)
        hann_window = hann_window if self.full_precision else hann_window.half()

        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('hann_window', hann_window)

    def mel_spectrogram(self, y: torch.Tensor) -> torch.Tensor:
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.ShortTensor) with shape (B, T) in range [-MIN_SHORT_INT, MAX_SHORT_INT]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        # convert short to float samples in range (-1; 1)
        float_type = torch.float32 if self.full_precision else torch.float16
        y = y.type(float_type) / 32768.0  # to range (-1, 1)

        padding = torch.zeros(y.size(0), int((self.n_fft - self.hop_size) / 2), device=y.device, dtype=y.dtype)
        y = torch.cat((padding, y, padding), dim=1)
        if self.full_precision:
            spec = torch.stft(
                y,
                self.n_fft,
                hop_length=self.hop_size,
                win_length=self.win_size,
                window=self.hann_window,
                center=self.center,
                pad_mode=self.pad_mode,
                normalized=False,
                onesided=True
            )
            spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        else:
            spec = self.stft(y)  # batch_size x fft_size/2 + 1 x frames

        spec = torch.matmul(self.mel_basis, spec)
        spec = self.spectral_normalize_torch(spec)
        if not self.full_precision and not self.skip_casts:
            # cast to fp32 before returning, so externally - no difference
            # if its half precision model or not
            spec = spec.type(torch.float32)
        return spec

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Forwards mel-spectrogram computation
        """
        return self.mel_spectrogram(y)

    def spectral_normalize_torch(self, magnitudes):
        output = self.dynamic_range_compression_torch(magnitudes)
        return output

    def dynamic_range_compression_torch(self, x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)
