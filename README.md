# 🚗 Sri Lankan License Plate Detection System 🇱🇰

> A high-performance real-time license plate detection system powered by YOLOv11, specifically trained and optimized for Sri Lankan license plates with both GPU and CPU support.

## ✨ Features

- 🎥 **Real-time Detection** - Live camera feed license plate recognition
- 🖼️ **Single Image Analysis** - Process individual images with high accuracy
- ⚡ **Performance Benchmarking** - Built-in CPU vs GPU performance comparison
- 📊 **System Monitoring** - Real-time GPU/CPU performance metrics
- 📸 **Screenshot Capture** - Save detection results instantly
- 💻 **Cross-platform Support** - Works on Windows, macOS, and Linux
- 🔧 **Flexible Inference** - Seamless switching between CPU and GPU processing

## 🎯 Demo

<table>
  <tr>
    <td align="center">
      <img width="300" alt="Real-time Detection" src="https://github.com/user-attachments/assets/1555585f-e371-41c0-a79d-7bdcecc0400b" />
      <br><b></b>
    </td>
    <td align="center">
      <img width="300" alt="Validation Results" src="https://github.com/user-attachments/assets/9517d066-5160-42ba-acd7-bd7559d8f471" />
      <br><b></b>
    </td>
  </tr>
</table>

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Camera device (for real-time detection)
- CUDA-compatible GPU (optional, but recommended)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/K-B-R-S-W/Realtime_License_Plate_Capture_System_Using_Yolo_V11.git
cd Realtime_License_Plate_Capture_System_Using_Yolo_V11

# Install dependencies
pip install -r requirements.txt

# Run the application
python "License Plate Recognition.py"
```

### Dependencies

```txt
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
```

## 📁 Project Structure

```
sri-lankan-license-plate-detection/
├── 📊 Charts and Predicts/            
├── 📁 Sample Outputs/                  
├── 🐍 License Plate Recognition.py    
├── 📓 License_Plate_Recognition_Data_Train.ipynb  
├── 🤖 best_model.pt                     
├── 📄 requirements.txt                
└── 📋 README.md                        
```

## 🚀 Usage

### Interactive Mode

Run the main script and choose from three available modes:

```bash
python "License Plate Recognition.py"
```

#### Available Modes:

1. **🎥 Real-time Camera Detection**
   - Live license plate detection from webcam
   - Real-time bounding box visualization
   - Performance metrics overlay

2. **🖼️ Single Image Detection**
   - Process individual image files
   - High-accuracy batch processing
   - Detailed detection analysis

3. **⚡ System Benchmark Test**
   - Performance comparison between CPU and GPU
   - Detailed timing analysis
   - Hardware utilization metrics

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `s` | Save screenshot |
| `i` | Show system information |
| `ESC` | Exit current mode |

## 🏋️‍♂️ Model Training

The YOLOv11 model was trained using state-of-the-art techniques on a curated Sri Lankan license plate dataset:

### Training Specifications

| Parameter | Value |
|-----------|-------|
| **Model Architecture** | YOLOv11s |
| **Training Images** | 921 samples |
| **Validation Images** | 77 samples |
| **Training Epochs** | 120 |
| **Input Resolution** | 640×640 pixels |
| **Batch Size** | 16 |
| **Optimizer** | AdamW |

### Training Pipeline

```python
# Training can be reproduced using the provided notebook
jupyter notebook License_Plate_Recognition_Data_Train.ipynb
```

## 📊 Performance Metrics

### Model Performance
- **mAP@0.5**: 95.2%
- **mAP@0.5:0.95**: 87.8%
- **Precision**: 94.1%
- **Recall**: 91.7%

### Hardware Performance
| Hardware | FPS | Inference Time |
|----------|-----|----------------|
| NVIDIA RTX 3050 | 30 FPS | ~6.9ms |
| Intel i5-13420h | 12 FPS | ~35.7ms |

## 🔧 Configuration

### GPU Setup (Optional)

For optimal performance with CUDA:

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Custom Model Training

To train with your own dataset:

1. Prepare your dataset in YOLO format
2. Update the configuration in the training notebook
3. Run the training pipeline
4. Replace `best_model.pt` with your trained model

### Development Setup

```bash
# Fork the repository
git fork https://github.com/K-B-R-S-W/Realtime_License_Plate_Capture_System_Using_Yolo_V11.git

# Create a feature branch
git checkout

# Make your changes and commit
git commit

# Push to your fork and submit a pull request
git push origin
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 🙏 Acknowledgments

- **[Ultralytics](https://github.com/ultralytics/ultralytics)** - For the incredible YOLOv11 framework
- **Sri Lankan License Plate Dataset Contributors** - For providing the training data
- **OpenCV Community** - For the powerful computer vision tools
- **PyTorch Team** - For the deep learning framework

---

## 📮 Support

**📧 Email:** [k.b.ravindusankalpaac@gmail.com](mailto:k.b.ravindusankalpaac@gmail.com)  
**🐞 Bug Reports:** [GitHub Issues](https://github.com/K-B-R-S-W/Realtime_License_Plate_Capture_System_Using_Yolo_V11/issues)  
**📚 Documentation:** See the project [Wiki](https://github.com/K-B-R-S-W/Realtime_License_Plate_Capture_System_Using_Yolo_V11/wiki)  
**💭 Discussions:** Join the [GitHub Discussions](https://github.com/K-B-R-S-W/Realtime_License_Plate_Capture_System_Using_Yolo_V11/discussions)

---

## ⭐ Support This Project

If you find this project helpful, please consider giving it a **⭐ star** on GitHub!
