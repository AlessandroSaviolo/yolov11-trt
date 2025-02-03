import os
import sys

from ultralytics import YOLO

#----------------------------------------------------------------------------

def main():
    print('Exporting YOLO11 to ONNX format.')
    print('Make sure to run this ON YOUR LAPTOP.')

    width = int(input(("Enter image width: ")))
    height = int(input(("Enter image height: ")))
    print(f"Image dimensions: {width}x{height}")
    input_shape = (height, width)

    checkpoints_path = "/".join(sys.path[0].split("/")[:-1]) + "/resources/checkpoints/"
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    model_type = str(input(("Enter model type [n, s, m, l, x]: ")))
    model_path = checkpoints_path + 'yolo11' + model_type + '.pt'
    model = YOLO(model_path)

    model.export(format="onnx", imgsz=input_shape, opset=11, nms=True)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
