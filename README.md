# YOLOv11 TensorRT
===========================

This repository hosts a C++ implementation of the state-of-the-art YOLOv11 object detection model from ultralytics, leveraging the TensorRT API for efficient, real-time inference.

### 1. Clone the Repository

```bash
git clone https://github.com/AlessandroSaviolo/yolov11-trt
cd yolov11-trt
```

### 2. Install Dependencies

- **For Python**:
  Install required Python dependencies using pip:
  
  ```bash
  pip install --upgrade ultralytics
  ```

- **For C++**:
  Ensure that OpenCV and TensorRT are installed. Set the correct paths for these libraries in the `CMakeLists.txt` file.

### 3. Build the C++ Code

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### 4. Exporting the Model

On your laptop:

1. Modify the `scripts/export.py` script if needed to set the desired model name.
2. Run the Python script to export the YOLOv11 model to ONNX format:

```bash
python scripts/export.py
```

### 5. Running Inference

On your robot:

Convert the ONNX model to a TensorRT engine:

```bash
./yolov11-trt ../resources/yolo11s.onnx ../assets/people.jpg
```

Perform object detection on an image:

```bash
./yolov11-trt ../resources/yolo11s.engine ../assets/people.jpg
```

## Acknowledgement

This code was forked from https://github.com/spacewalk01/yolov11-tensorrt and debugged and improved for easier readability.