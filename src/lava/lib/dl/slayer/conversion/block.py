# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import torch
import torch.nn.functional as F

from ..synapse import Conv
from ..utils.quantize import QuantizeAndClamp, MODE


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups=1, bias=False, activation=None,
                 num_msg_bits=16, msg_exp=6,
                 num_wgt_bits=8, wgt_exp=6,
                 num_scale_bits=16, scale_exp=12, scale=1.0) -> None:
        # Generic implementation of
        # y = f(scale * W * x + bias)
        super().__init__()

        self._quantized = False
        self._validate = False
        self.wgt_exp = wgt_exp
        self.num_da_bits = 24
        self.num_var_bits = 24
        self.input_quantizer = QuantizeAndClamp(num_bits=num_msg_bits,
                                                step=1 / (1 << msg_exp))
        self.wgt_quantizer = QuantizeAndClamp(num_bits=num_wgt_bits,
                                              step=1 / (1 << wgt_exp))
        self.da_quantizer = QuantizeAndClamp(num_bits=self.num_da_bits,
                                             step=1 / (1 << msg_exp + wgt_exp))
        self.bias_quantizer = QuantizeAndClamp(num_bits=self.num_var_bits,
                                               step=1 / (1 << msg_exp + wgt_exp))

        post_scale_exp = msg_exp + wgt_exp + num_scale_bits - scale_exp
        self.scale_quantizer = QuantizeAndClamp(num_bits=num_scale_bits + 1,
                                                step=1 / (1 << scale_exp))
        self.post_scale_quantizer = QuantizeAndClamp(num_bits=self.num_da_bits,
                                                     step=1 / (1 << post_scale_exp))
        self.scale = self.scale_quantizer(torch.tensor([scale]))
        self.scale_exp = scale_exp
        if self.scale == 1:
            self.scale = None  # No need to apply scale

        self.conv = Conv(in_channels, out_channels,
                         kernel_size, stride, padding, dilation,
                         groups, pre_hook_fx=self.wgt_quantizer)
        self.bias = None
        if bias:
            bias_shape = [1, out_channels, 1, 1, 1]
            self.bias = torch.nn.Parameter(torch.zeros(bias_shape))
        self.activation = activation

    def fixed_precision(self, validate=False):
        self._quantized = True
        self._validate = validate

    def full_precision(self):
        self._quantized = False

    def quantize(self, x: torch.tensor) -> torch.tensor:
        if torch.is_floating_point(x):
            x_int = (self.input_quantizer(x) /
                     self.input_quantizer.step).to(torch.int16)
        else:
            x_int = x.to(torch.int16)
            x = x.to(torch.float) * self.input_quantizer.step
        return x_int, x

    def dequantize(self, x: torch.tensor) -> torch.tensor:
        if torch.is_floating_point(x):
            return x
        else:
            return x.to(torch.float) * self.input_quantizer.step

    def forward(self, x):
        if self._quantized:
            return self.forward_quant(x)
        x = self.input_quantizer(x)
        z = self.conv(x)
        if self.scale is not None:
            scale = self.scale_quantizer(self.scale).to(z.device)
            z = self.post_scale_quantizer(z * scale)
        if self.bias is not None:
            z = z + self.bias_quantizer(self.bias)
        z = self.da_quantizer(z)
        if self.activation is not None:
            z = self.activation(z)
        z = self.input_quantizer(z, mode=MODE.FLOOR)
        return z

    def forward_quant(self, x):
        with torch.no_grad():
            x_int, x = self.quantize(x)
            if self._validate:
                x_gt = self.input_quantizer(x) / self.input_quantizer.step
                x_diff = x_int - x_gt
                if torch.abs(x_diff).max() > 0:
                    print('Validation WARNING: x error[int] = '
                          f'{torch.abs(x_diff).max()}')

            wgt = self.conv.weight.data
            weight_int = self.wgt_quantizer.quantize(wgt).to(torch.int8)
            if self._validate:
                weight_gt = self.wgt_quantizer(wgt) / self.wgt_quantizer.step
                wgt_diff = weight_int - weight_gt
                if torch.abs(wgt_diff).max() > 0:
                    print('Validation WARNING: wgt error[int] = '
                          f'{torch.abs(wgt_diff).max()}')

            bias_int = None
            if self.conv.bias is not None:
                bias_int = self.da_quantizer.quantize(
                    self.conv.bias.data).to(torch.int32)

            z_int = F.conv3d(
                x_int.to(torch.float64), weight_int.to(torch.float64),
                bias_int, self.conv.stride, self.conv.padding,
                self.conv.dilation, self.conv.groups,
            ).to(torch.int32)

            if self._validate:
                z = self.conv(self.input_quantizer.dequantize(x_int))
                z_diff = z_int - z / self.da_quantizer.step
                if torch.abs(z_diff).max() > 0:
                    print('Validation WARNING: w*x error[int'
                          f'] = {torch.abs(z_diff).max()}')

            if self.scale is not None:
                scale_int = self.scale_quantizer(
                    self.scale) / self.scale_quantizer.step
                z_int = (z_int * scale_int) >> self.scale_exp

            if self.bias is not None:
                bias_int = self.da_quantizer.quantize(
                    self.bias).to(torch.int32)
                z_int = z_int + bias_int
                num = 32 - self.num_var_bits
                z_int = (z_int << num) >> num
                if self._validate:
                    z = self.da_quantizer(z + self.bias_quantizer(self.bias))
                    z_diff = z_int - z / self.da_quantizer.step
                    if torch.abs(z_diff).max() > 0:
                        print('Validation WARNING: w*x + bias error[int] = '
                              f'{torch.abs(z_diff).max()}')

            if self.activation is not None:
                z_int = self.activation(z_int)
                if self._validate:
                    z = self.activation(z)
                    z_diff = z_int - z / self.da_quantizer.step
                    if torch.abs(z_diff).max() > 0:
                        print('Validation WARNING: f(w*x + bias) error[int] = '
                              f'{torch.abs(z_diff).max()}')

            z_int = (z_int >> self.wgt_exp).to(torch.int16)
            if self._validate:
                z = self.input_quantizer(
                    z, mode=MODE.FLOOR)
                z_diff = z_int - z / self.input_quantizer.step
                if torch.abs(z_diff).max() > 0:
                    print('Validation WARNING: y error[int] = '
                          f'{torch.abs(z_diff).max()}')

            return z_int
