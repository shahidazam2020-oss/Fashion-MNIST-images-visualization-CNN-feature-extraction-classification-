# 👕 Fashion-MNIST Image Classification Using CNN

## 📌 Project Overview

This project implements an **image classification system using a Convolutional Neural Network (CNN)** with **TensorFlow and Keras**. The model is trained on the **Fashion-MNIST dataset**, which contains grayscale images representing different types of clothing and fashion items.

The project demonstrates the complete machine learning workflow, including **dataset loading, image visualization, class distribution analysis, image normalization, CNN model development, model training, learning-curve analysis, prediction, and confusion-matrix evaluation**.

---

## 🎯 Project Objective

**The main objective of this project is to develop a deep learning model capable of automatically recognizing and classifying fashion images into their corresponding categories.**

The model classifies images into **10 different categories**:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

**These class names are explicitly defined in the project notebook.**

---

## 📊 Dataset

The project uses the **Fashion-MNIST dataset**, loaded directly through TensorFlow/Keras.

```python
from tensorflow.keras.datasets import fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

The images are grayscale and have a resolution of **28 × 28 pixels**.

## The notebook also visualizes randomly selected images and examines the distribution of images across the different classes.

## 🧹 Data Preprocessing

Before training the CNN, the image pixel values are normalized by dividing them by **255.0**.

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

**This converts the pixel values from the original 0–255 range to a normalized range between 0 and 1.**

**The notebook also performs a second normalized-model experiment to compare the training and validation performance.**

---

## 🧠 CNN Model Architecture

The project uses a **Convolutional Neural Network (CNN)** implemented with TensorFlow/Keras.

**The architecture consists of:**

* **Conv2D layer** with 32 filters
* **MaxPooling2D layer**
* **Conv2D layer** with 64 filters
* **MaxPooling2D layer**
* **Conv2D layer** with 64 filters
* **Flatten layer**
* **Dense layer** with 64 neurons
* **Output Dense layer** with 10 neurons
* **ReLU activation** for hidden layers
* **Softmax activation** for the final classification layer

The model is compiled using the **Adam optimizer**, sparse categorical cross-entropy loss, and accuracy as the evaluation metric.

The model is trained for **10 epochs** using the training dataset and validation data.

---

## 📈 Model Training

The CNN is trained using:

```python
model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)
```

**During training, the notebook stores the training history, including:**

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss

**This information is later used to visualize the learning curves.**

---

## 📉 Learning Curve Analysis

**The project generates learning curves to monitor model performance during training.**

Two major plots are produced:

### Training & Validation Loss

This plot helps analyze how the model's loss changes across training epochs.

### Training & Validation Accuracy

This plot shows how accurately the model performs on the training and validation datasets throughout training.

The notebook includes a dedicated function for visualizing these learning curves.

---

## 🔍 Model Prediction

**After training, the CNN is used to predict the classes of the test images.**

**The predicted probabilities are converted into predicted class labels using `argmax`.**

```python
test_predictions = model.predict(test_images)
test_predictions_labels = np.argmax(test_predictions, axis=1)
```

**The predicted labels are then compared with the actual test labels.**

---

## 📊 Confusion Matrix

A **confusion matrix** is generated to evaluate the classification performance of the CNN.

It provides a detailed view of:

* Correctly classified images
* Incorrectly classified images
* Confusion between different fashion categories

**The project uses a heatmap to visualize the confusion matrix, with the 10 Fashion-MNIST classes displayed on both axes.**

**This is particularly useful for identifying categories that the CNN finds difficult to distinguish, such as visually similar clothing items.**

---

## 🔄 Model Comparison

**The notebook also creates a second CNN model using the normalized image data.**

**The performance of the original and normalized models is compared using:**

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss

**The project therefore demonstrates how preprocessing can be evaluated by comparing model learning behavior and performance.**

---

## 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Convolutional Neural Networks (CNN)**
* **Jupyter Notebook**

**The notebook imports TensorFlow, NumPy, Scikit-learn's confusion-matrix functionality, Seaborn, and Matplotlib.**

---

## 💡 Use of This Project

**This software/project can be used for:**

* 👕 Automatic clothing image classification
* 🤖 Demonstrating computer vision using deep learning
* 🧠 Learning CNN architecture and image classification
* 📊 Evaluating multiclass classification models
* 🎓 Academic machine learning and deep learning projects
* 🔬 Experimenting with image preprocessing techniques
* 📚 Understanding model training and validation
* 🖼️ Building a foundation for more advanced image-recognition applications

---

## ⭐ Key Features

* ✅ Fashion-MNIST dataset loading
* ✅ Random image visualization
* ✅ Class distribution visualization
* ✅ Image normalization
* ✅ CNN-based image classification
* ✅ 10-class fashion recognition
* ✅ Model training for 10 epochs
* ✅ Training and validation accuracy analysis
* ✅ Training and validation loss analysis
* ✅ Test-set prediction
* ✅ Confusion matrix visualization
* ✅ Comparison of model performance

---

## 📁 Project Structure

```text
Fashion-MNIST/
│
├── Fashion MNIST.ipynb
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Install Required Libraries

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### 3. Open the Notebook

Open:

```text
Fashion MNIST.ipynb
```

using **Jupyter Notebook**, **JupyterLab**, or **Google Colab**.

### 4. Run the Cells

**Execute the notebook cells sequentially to:**

1. Load Fashion-MNIST
2. Visualize sample images
3. Analyze class distribution
4. Normalize the images
5. Build the CNN
6. Train the model
7. Plot learning curves
8. Generate predictions
9. Create the confusion matrix
10. Compare model performance

---

## 🎓 Learning Outcomes

**This project demonstrates practical experience with:**

* Deep learning
* Convolutional Neural Networks
* Image preprocessing
* Multiclass classification
* TensorFlow/Keras
* Model training and validation
* Performance visualization
* Confusion-matrix analysis
* Computer vision fundamentals

---

## 👨‍💻 About the Developer

**Shahid Azam**

**MS Computer Science Student**
**Specialization in Artificial Intelligence**

Institute of Management Sciences, Peshawar, Pakistan

I enjoy building complete software systems—from networking and operating system concepts to backend architecture, machine learning, and distributed systems.
