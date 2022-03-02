import caffe
import numpy as np
import pandas as pd
import argparse
import os


def extract_caffe_model(model, weights, output_path):
  """extract caffe model's parameters to numpy array, and write them to files
  Args:
    model: path of '.prototxt'
    weights: path of '.caffemodel'
    output_path: output path of numpy params 
  Returns:
    None
  """
  net = caffe.Net(model, caffe.TEST)
  net.copy_from(weights)

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  for item in net.params.items():
    name, layer = item
    #print('convert layer: ' + name)

    num = 0
    for p in net.params[name]:
      name = name.replace('/','-')
      if name.find("bn")<=0 and name.find("conv")==0:
          np.save(output_path + '/' + str(name), p.data)
          data = p.data.reshape(p.data.shape[0], p.data.shape[1]* p.data.shape[2]* p.data.shape[3])
          for idx in range(data.shape[0]):
            data3 = list(map(abs,data[idx]))
            print(name, idx, sum(data3))
          print("\n")



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", help="model prototxt path .prototxt")
  parser.add_argument("--weights", help="caffe model weights path .caffemodel")
  parser.add_argument("--output", help="output path")
  args = parser.parse_args()
  extract_caffe_model(args.model, args.weights, args.output)