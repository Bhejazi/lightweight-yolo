# Single Class Detector
### A lightweight object detection pipeline to draw bounding boxes around a single class of objects in images.

Keras / TensorFlow 2.x Framework built on a MobileNetV2 (pretrained on ImageNet) model

Utilizing a simple YOLO-style grid (default 7×7) predicting per-cell [objectness, x, y, w, h] (all normalized to [0,1])


## Brief description of repository components:

model/detector.py –> Model architecture (MobileNetV2 backbone + YOLO-style head) and custom loss

utils/labels.py –> Parse YOLO labels and build (S×S×5) training targets

utils/data.py –> Discover dataset files and build a tf.data pipeline

train.py –> CLI training script

inference.py –> CLI inference & visualization on new images


# Setup
### 1. Download or clone the repository

### 2. Setup environment using Conda
```
conda create --name macaque-env python=3.10
conda activate macaque-env
```

### 3. Install required libraries
```
python -m pip install -r requirements.txt
```

### 4. Train the model

i. Activate your environment and cd into the working directory of the code

ii. Single line using Anaconda Prompt to train model:
```
python train.py --data_dir X:/YourDataDirectory --epochs 1 --batch_size 8 --output_dir outputs
```

### 5. Run Inference on a New Image
Single line using Anaconda Prompt to train model:
```
python inference.py --model_path outputs\detector.keras --image_path X:/YourImgDirectory/img.jpg --save_path outputs\inference_result.jpg --score_threshold 0.5 --grid_size 7 --input_size 224
```
