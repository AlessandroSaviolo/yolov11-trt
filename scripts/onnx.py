import sys
import os
import numpy as np
import torch
import pytorch_lightning
from onnx import checker, load

from ultralytics import YOLO

#----------------------------------------------------------------------------

class ObjectDetectionModel(pytorch_lightning.LightningModule):
    def __init__(self, model_path, device):
        super().__init__()
        self.model = YOLO(model_path)

    def forward(self, inp):
        _, out = self.model(inp)
        return out

#----------------------------------------------------------------------------

def export_model_to_onnx(model_torch_path, model_onnx_path, input_shape):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Found device', device)

    pytorch_lightning.seed_everything(17)
    sample_input = torch.rand(1, 3, input_shape[0], input_shape[1]).float().to(device)
    
    model = ObjectDetectionModel(model_torch_path, device)
    model = model.eval().to(device)

    # try:
    #     model.to_onnx(
    #         model_onnx_path,
    #         sample_input,
    #         export_params=True,
    #         opset_version=11,
    #         do_constant_folding=True,
    #         input_names=['input'],
    #         output_names=['output'],
    #         dynamic_axes={
    #             'input':  {0: 'batch_size'}, 
    #             'output': {0: 'batch_size'}
    #         }
    #     )
    #     checker.check_model(load(model_onnx_path))
    #     print('ONNX model export and validation successful.')
    # except Exception as e:
    #     print(f'Error during model export or validation: {e}')

#----------------------------------------------------------------------------

def get_multiple_of_14(prompt):
    while True:
      try:
          value = int(input(prompt))
          if value % 14 == 0:
              return value
          else:
              print("Value must be a multiple of 14. Please try again.")
      except ValueError:
          print("Invalid input. Please enter an integer.")

#----------------------------------------------------------------------------

def main():
    print('Exporting YOLO v11 to ONNX format.')
    print('Make sure to run this ON YOUR LAPTOP.')

    checkpoints_path = "/".join(sys.path[0].split("/")[:-1]) + "/resources/checkpoints/"
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    # width = get_multiple_of_14("Enter image width (multiple of 14): ")
    # height = get_multiple_of_14("Enter image height (multiple of 14): ")
    # print(f"Image dimensions: {width}x{height}")
    # input_shape = (height, width)

    # model_type = str(input(("Enter model type [n, s, m, l, x]: ")))

    input_shape = (256, 320)
    model_type = 's'

    model_torch_path = checkpoints_path + 'yolo11' + model_type + '.pt'
    model_onnx_path = model_torch_path + 'yolo11' + model_type + '.onnx'

    export_model_to_onnx(model_torch_path, model_onnx_path, input_shape)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()