# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Quantization utility."""

import torch
from enum import IntEnum, unique


@unique
class MODE(IntEnum):
    """Quantization mode constants. Options are {``ROUND : 0``, ``FLOOR : 1``}.
    """
    ROUND = 0
    FLOOR = 1


class _quantize(torch.autograd.Function):
    """ """
    @staticmethod
    def forward(ctx, input, step=1):
        """
        """
        # return input
        # print('input quantized with step', step)
        return torch.round(input / step) * step

    @staticmethod
    def backward(ctx, gradOutput):
        """
        """
        return gradOutput, None


class _floor(torch.autograd.Function):
    """ """
    @staticmethod
    def forward(ctx, input, step=1):
        """
        """
        # return input
        # print('input quantized with step', step)
        return torch.floor(input / step) * step

    @staticmethod
    def backward(ctx, gradOutput):
        """
        """
        return gradOutput, None


def quantize(input, step=1, mode=MODE.ROUND):
    """Implements quantization of parameters. Round or floor behavior can be
    selected using mode argument.

    Parameters
    ----------
    input : torch tensor
        input tensor
    step : float
        quantization step. Default is 1.
    mode : MODE
        quantization mode. Default is MODE.ROUND.

    Returns
    -------
    torch tensor
        quantized tensor

    Examples
    --------

    >>> # Quantize in step of 0.5
    >>> x_quantized = quantize(x, step=0.5)
    """
    if mode == MODE.ROUND:
        return _quantize.apply(input, step)
    elif mode == MODE.FLOOR:
        return _floor.apply(input, step)
    else:
        raise ValueError(f'{mode=} is not recognized.')


def quantize_hook_fx(x: torch.tensor,
                     scale: int = (1 << 6),
                     num_bits: int = 8,
                     descale: bool = False) -> torch.tensor:
    """Quantize prehook function to use in slayer synapse pre-hook for
    quantization.

    Parameters
    ----------
    x : torch.tensor
        Input tensor.
    scale : int, optional
        Quantization decimal scale corresponding to 1.0 value,
        by default (1 << 6).
    num_bits : int, optional
        Number of bits to use in quantization, by default 8.
    descale : bool, optional
        Flag to descale the fixed point number to integer or keep it as
        fixed point number. By default False.

    Returns
    -------
    torch.tensor
        Quantized tensor.
    """
    min = -2 * (1 << num_bits)
    max = 2 * ((1 << num_bits) - 1)
    if descale is False:
        return quantize(x, step=2 / scale).clamp(min / scale, max / scale)
    else:
        return quantize(x, step=2 / scale).clamp(min / scale,
                                                 max / scale) * scale


class QuantizeAndClamp:
    def __init__(self,
                 num_bits: int,
                 step: float,
                 quant_mode: MODE = MODE.ROUND) -> None:
        """Quantize and clamp a tensor.

        Parameters
        ----------
        num_bits : int
            Number of bits to represent the tensor in fixed point.
        step : float
            Quantization step
        quant_mode : MODE, optional
            Mode of quantization as described by MODE enum. By default
            MODE.ROUND.
        """
        self.num_bits = num_bits
        self.step = step
        self.amax = ((1 << num_bits - 1) - 1) * self.step
        self.quant_mode = quant_mode

    def __call__(self,
                 x: torch.tensor,
                 mode: MODE | None = None) -> torch.tensor:
        """When called, it performs fake quantization with passthrough gradient.

        Parameters
        ----------
        x : torch.tensor
            Input tensor.
        mode : MODE, optional
            Mode of quantization to use for this particular call. If none,
            the quantizer object's quantization mode is used, by default None.

        Returns
        -------
        torch.tensor
            Output tessor with quantization step applied.
        """
        if mode is None:
            mode = self.quant_mode
        return quantize(x,
                        step=self.step,
                        mode=mode).clamp(-self.amax, self.amax)

    def quantize(self, x: torch.tensor) -> torch.tensor:
        """Quantize a tensor to integer values. Note: it does not change the
        data type.

        Parameters
        ----------
        x : torch.tensor
            Input tensor in full precision.

        Returns
        -------
        torch.tensor
            Quantized tensor in fixed precision.
        """
        return self(x) / self.step

    def dequantize(self, x: torch.tensor) -> torch.tensor:
        """Dequantize a tensor to full precision value.

        Parameters
        ----------
        x : torch.tensor
            Input tensor in fixed precision.

        Returns
        -------
        torch.tensor
            Output tensor in full precision.
        """
        return x * self.step
