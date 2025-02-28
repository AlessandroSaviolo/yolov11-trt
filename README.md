# YOLOv11 TensorRT
===========================

This repository hosts a C++ implementation of the state-of-the-art YOLOv11 object detection model from ultralytics, leveraging the TensorRT API for efficient, real-time inference.

### 1. Clone the Repository

```bash
git clone https://github.com/AlessandroSaviolo/yolov11_trt
cd yolov11_trt
```

### 2. Install Dependencies

- **For Python**:
  Install required Python dependencies using pip:
  
  ```bash
  pip install --upgrade ultralytics
  ```

- **For C++**:
  Ensure that OpenCV and TensorRT are installed. Set the correct paths for these libraries in the `CMakeLists.txt` file.

### 3. Exporting the Model from PYTORCH to ONNX

1. Modify the `scripts/export.py` script if needed to set the desired model name.
2. Run the Python script to export the YOLOv11 model to ONNX format:

```bash
cd ~/yolov11_trt
python scripts/export_onnx.py
```

### 4. Build the C++ Code

```bash
cd ~/yolov11_trt
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### 5. Exporting the Model from ONNX to TensorRT

Move the generated `yolo11s.onnx` file into the `checkpoints` folder:
```bash
cd ~/yolov11_trt
mkdir checkpoints
mv yolo11s.onnx checkpoints
```

Convert the ONNX model to a TensorRT engine:

```bash
cd ~/yolov11_trt/build
./yolov11_trt ../checkpoints/yolo11s.onnx ../assets/people.jpg
```

### 6. Running Inference

Perform object detection on a sample image:

```bash
./yolov11_trt ../checkpoints/yolo11s.engine ../assets/people.jpg
```

## Acknowledgement

This code was forked from https://github.com/spacewalk01/yolov11-tensorrt and debugged and improved for easier readability.
