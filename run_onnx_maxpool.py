#!/usr/bin/env python3

import onnx

import onnxruntime
from onnx import optimizer
import numpy as np
import onnx.helper
import caffe2.python.onnx.backend as c2

model_path = "test_maxpool.onnx"
print(model_path)

input_data = np.random.rand(1, 1, 64, 64).astype(np.float32)

model = onnx.load(model_path)


input_names = ["x"]
x_c2 = input_data
#x_c2 = np.transpose(input_data, [0, 2, 3, 1])
y = np.expand_dims(x_c2, axis=0)
caffe_res = c2.run_model(model, dict(zip(input_names, y)))[0]
print(caffe_res)

