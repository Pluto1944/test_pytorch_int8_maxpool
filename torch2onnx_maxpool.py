#!/usr/bin/env python
# coding=utf-8
import os
import sys
import time
import torch
import torch.nn as nn

import numpy as np
import io

import onnx

from torch.quantization import QuantStub, DeQuantStub
import torch.quantization as quantizer

torch.backends.quantized.engine = "qnnpack"

def export_to_onnx(model, input, input_names):
    outputs = model(input)

    traced = torch.jit.trace(model, input)
    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    buf.seek(0)

    model = torch.jit.load(buf)
    f = io.BytesIO()
    torch.onnx.export(model, input, "test_maxpool.onnx", input_names=input_names, example_outputs=outputs,
                      verbose=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    f.seek(0)

    onnx_model = onnx.load(f)
    return onnx_model

class ConvModel(torch.nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.quant = QuantStub()
        self.conv_0 = nn.Conv2d(1, 48, kernel_size=7, stride=2, padding=3, bias=True)
        self.relu_0 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode = True)
        #self.conv_1 = nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1, bias=True)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv_0(x)
        x = self.relu_0(x)
        x = self.maxpool(x)
        #x = self.conv_1(x)
        x = self.dequant(x)
        return x

model = ConvModel().cpu()

quantizer.fuse_modules(model, ['conv_0', 'relu_0'], inplace=True)

qconfig = torch.quantization.default_qconfig
model.qconfig = qconfig
model = torch.quantization.prepare(model)
model = torch.quantization.convert(model.eval())

x_numpy = np.random.rand(1, 1, 64, 64).astype(np.float32)
x = torch.from_numpy(x_numpy).to(dtype=torch.float)
outputs = model(x)
input_names = ["x"]
onnx_model = export_to_onnx(model, x, input_names)
print("finish!")
