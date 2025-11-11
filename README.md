# AI-Powered-Image-Classification-System
A custom-built 3-layer CNN using TensorFlow/Keras to classify images from the CIFAR-10 dataset. This project includes the full data pipeline, model training, and evaluation notebook.

**🤖 AI-Powered Image Classification System (CIFAR-10)**
This project is a complete, end-to-end deep learning pipeline built to classify images from the CIFAR-10 dataset. It fulfills the requirements of the Flikt Technology AI Developer Assignment by demonstrating a full workflow from data preprocessing to model training and evaluation.

This repository contains the complete source code, the final trained model, and all evaluation results.

**🚀 How to Download and Run This Project**
You can replicate this project and run the code on your own machine by following these steps.

**Prerequisites**
Python 3.8+
Git (for cloning)

**Step 1: Get the Code (Clone the Repository)**
Open your terminal or command prompt and run the following command to clone this repository to your local machine:

git clone https://github.com/[Your-Username]/[Your-Repo-Name].git

cd [Your-Repo-Name]

(Replace [Your-Username] and [Your-Repo-Name] with your actual GitHub details)

**Step 2: Install Dependencies**
All the required Python libraries are listed in the requirements.txt file. Install them using pip:

pip install -r requirements.txt

**Step 3: Run the Jupyter Notebook**
The entire project is contained within a single Jupyter Notebook. To start the server, run:

python -m notebook

This will automatically open a new tab in your web browser. From the file list, click on training_notebook.ipynb to open it.

**You can then run each cell in the notebook from top to bottom to:**

Load and preprocess the data.
Build the CNN model.
Train the model (this will take time).
Evaluate the model and see all the final graphs and reports.

**🛠️ Technologies Used**

This project was built using a standard, professional data science stack:

#Python: The core programming language for the entire project.

#TensorFlow (with Keras): The deep learning framework used to design, compile, and train the custom Convolutional Neural Network.

#Jupyter Notebook: The interactive environment used for all development, from data exploration to training and final evaluation.

#Scikit-learn (sklearn): Used for splitting the data (the 70/15/15 split) and for generating the final evaluation metrics (the Classification Report and Confusion Matrix).

#NumPy: The fundamental library for all numerical operations and for handling the image data as arrays.

#Matplotlib & Seaborn: Used to create all the data visualizations, including the training/loss curves and the confusion matrix heatmap.

#Git & GitHub: Used for version control and for hosting the final project.

**📁 Project File Structure**
Here's what each file in this repository does:

training_notebook.ipynb: (Source Code) This is the main file containing all the Python code for data loading, preprocessing, model building, training, and evaluation.

basic_cnn_model.h5: (Trained Model File) This is the final, best-performing model that was saved after training.

README.md: (Project Report) This file, which you are currently reading.
requirements.txt: A list of all Python libraries needed to run the project.

training_curves.png: The saved graph of the model's accuracy and loss.

confusion_matrix.png: The saved heatmap of the model's predictions.
sample_predictions.png: The saved image of sample predictions.

**🧠 Model Architecture & Performance**
**1. Data Pipeline**
Dataset: We used CIFAR-10, a standard benchmark dataset of 60,000 32x32 color images.

Splitting: The data was split into the required 70% / 15% / 15% ratio:

Training Set: 42,000 images

Validation Set: 9,000 images

Testing Set: 9,000 images

Normalization: All pixel values were scaled from 0-255 to 0.0-1.0 for faster and more stable training.

**2. CNN Model Architecture**
As required, I built a custom CNN from scratch with 3 convolutional layers:

Input: (32, 32, 3)

Block 1: Conv2D(32 filters) -> BatchNormalization -> MaxPooling

Block 2: Conv2D(64 filters) -> BatchNormalization -> MaxPooling

Block 3: Conv2D(128 filters) -> BatchNormalization -> MaxPooling

Classifier: Flatten -> Dense(128) -> Dropout(0.5) -> Dense(10, 'softmax')

This architecture uses BatchNormalization to stabilize learning and Dropout to prevent overfitting.

**3. Final Test Results**
The model was evaluated on the 9,000-image test set, which it had never seen before.

Final Test Accuracy: 75.02%

Final Test Loss: 0.7605

**📊 All Project Visuals**
Training & Validation Curves
These graphs show the model's accuracy and loss as it trained. You can see the "Train Accuracy" (blue) and "Validation Accuracy" (orange) both increase and stay close, showing the model learned well without just memorizing.

<img width="1071" height="444" alt="image" src="https://github.com/user-attachments/assets/dab3af88-7cc6-44e0-a175-46b2cf97e73a" />


**Classification Report**
This report shows the performance for each class. The model is very good at identifying automobiles (90% precision) and ships (87% precision) but struggles with cats (52% precision), which it often confuses with dogs.

<img width="890" height="612" alt="image" src="https://github.com/user-attachments/assets/91ac3ff6-0889-42f6-83b6-fb20752e960f" />


**Confusion Matrix**

This heatmap visualizes the model's mistakes. The dark blue diagonal line shows all the correct guesses. The other numbers show the errors (e.g., it mistook 149 cats for dogs).
