# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import torch

class Network(torch.nn.Module):
    def __init__(self, store_all_buffer=False):
        super().__init__()
        self.blocks = torch.nn.ModuleList([])
        self.blk_inps = []
        self.blk_save = None
        self.buffer = {}  # only store things that are necessary
        self.input_idx = []
        self.output_idx = []
        self.store_all_buffer = store_all_buffer
        self._quantized = False

    def fixed_precision(self, validate=True):
        self._quantized = True
        for block in self.blocks:
            if hasattr(block, 'fixed_precision'):
                block.fixed_precision(validate=validate)

    def full_precision(self):
        self._quantized = False
        for block in self.blocks:
            if hasattr(block, 'full_precision'):
                block.full_precision()

    def add_block(self, block, inp_idx=-1):
        self.blocks.append(block)
        self.blk_inps.append(inp_idx)

    def setup_buffer(self):
        self.blk_save = [False] * len(self.blocks)
        for idx, blk_inp in enumerate(self.blk_inps):
            if blk_inp != -1:
                self.blk_save[idx + blk_inp] = True
        for idx in self.output_idx:
            self.blk_save[idx] = True

    def forward_buffer(self):
        y = None
        if self.blk_save is None:
            self.setup_buffer()
        for idx, (blk, blk_inp) in enumerate(zip(self.blocks, self.blk_inps)):
            if y is None or blk_inp != -1:
                x = self.buffer[idx + blk_inp]
            else:
                x = y
            y = blk(x)
            # print(f'{x.dtype = }, {y.dtype = }')
            if self.blk_save[idx] or self.store_all_buffer:
                self.buffer[idx] = y.clone()

    def forward(self, *inputs):
        input_depth = - len(inputs)
        for idx, inp in enumerate(inputs):
            self.buffer[input_depth + idx] = inp
        self.forward_buffer()

        if len(self.output_idx) == 1:
            out_idx = self.output_idx[0]
            out = self.buffer[out_idx]
            if hasattr(self.blocks[out_idx], 'fixed_precision'):
                out = self.blocks[out_idx].dequantize(out)
            return out
        elif len(self.output_idx) > 1:
            outputs = []
            for idx in self.output_idx:
                output = self.buffer[idx]
                if hasattr(self.blocks[idx], 'fixed_precision'):
                    outputs.append(self.blocks[idx].dequantize(output))
                else:
                    outputs.append(output)
            return (*outputs, )
