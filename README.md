# 🌾 Rice Leaf Disease Detection Using MobileNetV2 and Flask

A deep learning-based web application for detecting diseases in rice leaves using **MobileNetV2** and **Flask**.

The application allows users to upload a rice leaf image through a web interface and predicts the disease category along with the prediction confidence.

## 📌 Project Overview

Rice is one of the most important food crops, and diseases affecting rice leaves can significantly reduce crop production.

This project uses **Transfer Learning with MobileNetV2** to classify rice leaf images into three disease categories:

* 🌿 Bacterial Leaf Blight
* 🍂 Brown Spot
* 🌱 Leaf Smut

The trained deep learning model is deployed using **Flask**, allowing users to perform predictions through a web browser.

## 🎯 Objectives

* Detect diseases from rice leaf images.
* Build a deep learning image classification model.
* Use MobileNetV2 transfer learning for classification.
* Preprocess images before prediction.
* Save the trained model for deployment.
* Develop a Flask-based web application.
* Display the predicted disease and confidence score.

## 🧠 Model Used

### MobileNetV2

MobileNetV2 is a lightweight convolutional neural network architecture designed for efficient image classification and computer vision applications.

In this project, the pretrained MobileNetV2 model is used as the feature extractor.

The base model is frozen and a custom classification layer is added on top.

### Model Architecture

```text
Input Image
     │
     ▼
224 × 224 × 3
     │
     ▼
MobileNetV2
Pretrained on ImageNet
     │
     ▼
Global Average Pooling
     │
     ▼
Dense Layer
     │
     ▼
Softmax
     │
     ▼
3 Disease Classes
```

The project uses a `224 × 224 × 3` input size and rescales image pixel values using `1./255`. This matches the preprocessing used during model training.

## 📊 Dataset

The project dataset contains rice leaf images belonging to three disease classes:

| Disease               | Class |
| --------------------- | ----: |
| Bacterial Leaf Blight |     0 |
| Brown Spot            |     1 |
| Leaf Smut             |     2 |

The original project contains **119 images** distributed across the three classes.

For model training, an **80/20 training-validation split** is used.

```text
Total Images
     │
     ├── Training Data → 80%
     │
     └── Validation Data → 20%
```

## 🔄 Image Preprocessing

Before training and prediction, images are:

1. Loaded from the dataset.
2. Resized to `224 × 224`.
3. Converted to RGB format.
4. Pixel values are scaled using:

```python
image = image / 255.0
```

The same preprocessing is applied during Flask prediction to ensure consistency between training and deployment.

## 🚀 Model Training

The MobileNetV2 base model is initialized with ImageNet weights:

```python
base = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base.trainable = False
```

A Global Average Pooling layer and classification layer are added:

```python
x = GlobalAveragePooling2D()(base.output)

output = Dense(
    3,
    activation="softmax"
)(x)
```

The model is compiled using:

```python
model.compile(
    optimizer=Adam(0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

Early stopping is used during training to prevent unnecessary training once validation performance stops improving.

## 📈 Model Performance

The MobileNetV2 model achieved approximately:

```text
Validation Accuracy: 91.30%
Validation Loss:     0.4307
```

The reported result is based on the validation evaluation from the project notebook.

> Note: The dataset is relatively small, so the reported accuracy should not be interpreted as production-level generalization performance. Testing with a larger and more diverse dataset would provide a stronger evaluation.

## 🌐 Flask Web Application

The trained model is deployed using Flask.

### Application Workflow

```text
User
  │
  ▼
Open Flask Web Application
  │
  ▼
Upload Rice Leaf Image
  │
  ▼
Image Validation
  │
  ▼
Resize to 224 × 224
  │
  ▼
Normalize Pixel Values
  │
  ▼
MobileNetV2 Model
  │
  ▼
Prediction
  │
  ▼
Disease + Confidence
```

## 📁 Project Structure

```text
rice-leaf-disease-detection-flask/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
|   └── PRCP-1001-RiceLeaf.ipynb
│   └── rice_leaf_mobilenetv2.keras
│
├── Rice_Leaf_Disease_Dataset/
│   ├── Bacterial leaf blight/
│   ├── Brown spot/
│   └── Leaf smut/
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   └── uploads/
│
└── templates/
    ├── index.html
    └── result.html
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Manojg14/rice-leaf-disease-detection.git
```

```bash
cd rice-leaf-disease-detection
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🧪 Train the Model

If you want to train the model yourself, make sure the dataset follows this structure:

```text
Rice_Leaf_Disease_Dataset/
│
├── Bacterial leaf blight/
├── Brown spot/
└── Leaf smut/
````

After training, the model will be saved as:

```text
model/rice_leaf_mobilenetv2.keras
```

## ▶️ Run the Flask Application

Start the Flask server:

```bash
python app.py
```

The application will run at:

```text
http://127.0.0.1:5000
```

Open the URL in your browser.

## 🔍 Prediction Example

The user uploads a rice leaf image:

```text
        Rice Leaf Image
              │
              ▼
       MobileNetV2 Model
              │
              ▼
       Predicted Disease
              │
              ▼
    Bacterial Leaf Blight
              │
              ▼
       Confidence: 94.32%
```

The actual confidence value will depend on the uploaded image and model prediction.

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning / Deep Learning

* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Pillow

### Web Development

* Flask
* HTML
* CSS

### Development Environment

* Jupyter Notebook
* VS Code
* Git
* GitHub

## 📦 Requirements

Main dependencies:

```text
Flask
TensorFlow
NumPy
Pillow
Werkzeug
```

Install them using:

```bash
pip install -r requirements.txt
```

## 🔐 File Upload Security

The Flask application validates uploaded files and allows only supported image formats:

```text
.jpg
.jpeg
.png
.webp
.jfif
```

Uploaded filenames are processed using `secure_filename()` before saving them.

## ⚠️ Limitations

* The dataset is relatively small.
* Real-world rice leaf images can contain different lighting, backgrounds and camera angles.
* Model performance may vary on images outside the training distribution.
* The current application focuses on three disease categories.
* A larger dataset would help improve model generalization.

## 🔮 Future Improvements

Possible improvements include:

* Increase the size of the training dataset.
* Add more rice diseases.
* Use data augmentation.
* Fine-tune MobileNetV2 layers.
* Add Grad-CAM visualization.
* Add disease treatment recommendations.
* Add prediction history.
* Add database integration.
* Deploy the Flask application to AWS.
* Add REST API endpoints.
* Add authentication for users.
* Improve the frontend UI.

## 📌 Project Workflow

```text
Dataset Collection
       ↓
Data Exploration
       ↓
Image Preprocessing
       ↓
Train / Validation Split
       ↓
MobileNetV2 Transfer Learning
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Save Trained Model
       ↓
Flask Integration
       ↓
Image Upload
       ↓
Disease Prediction
       ↓
Display Result
```

## 📊 Results

The project compared a custom CNN and MobileNetV2.

| Model       | Validation Accuracy |
| ----------- | ------------------: |
| CNN         |             ~52.17% |
| MobileNetV2 |             ~91.30% |

MobileNetV2 performed substantially better than the custom CNN in the project evaluation.

## 👨‍💻 Author

**Manoj G**

AI / Machine Learning Developer

GitHub: `https://github.com/Manoj14`
