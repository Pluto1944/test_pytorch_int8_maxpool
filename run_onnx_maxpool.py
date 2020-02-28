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

#input_data = np.loadtxt("./input_224x224.txt")
#input_data = input_data.reshape(1, 3, 224, 224)
model = onnx.load(model_path)


input_names = ["x"]
x_c2 = input_data
#x_c2 = np.transpose(input_data, [0, 2, 3, 1])
y = np.expand_dims(x_c2, axis=0)
caffe_res = c2.run_model(model, dict(zip(input_names, y)))[0]
print(caffe_res)

'''
onnx.checker.check_model(model)
print("finish")
sess = onnxruntime.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
pred = sess.run([output_name], {input_name: input_data.astype(np.float32)})
print(pred[0])
print("the output_name:{}:{}".format(output_name, np.shape(pred[0])))
#np.savetxt("./output-onnx.txt", pred[0].reshape(-1), fmt="%0.18f")
'''
