# Brain Tumor Detection Project
![Uploading image.png…]()

## Overview

This project is a deep learning-based application for detecting brain tumors from MRI images. It uses a pre-trained MobileNetV2 model with transfer learning and Grad-CAM visualization to highlight regions of interest in the images.

## Features
- Preprocessing of MRI images.
- Tumor detection using deep learning models.
- Classification of tumor types.
- Visualization of results.
- Grad-CAM Visualization: Generates heatmaps to visualize the regions of the image 
  that contributed to the model's decision.
- Interactive Web App: Built using Streamlit for easy interaction and visualization.

## Project Structure
- `brain_tumor_dataset/`: Contains the dataset of MRI images categorized into `no` (no tumor) and `yes` (tumor).
- `app.py`: Streamlit-based web application for uploading MRI images and visualizing predictions with Grad-CAM.
- `model.py`: Script for training the deep learning model using MobileNetV2 and saving the trained model.
- `brain_tumor_model.h5`: Saved trained model.
- `history.pkl`: Training history for visualization and analysis.

## Usage 
1. Train the Model
To train the model, run the model.py script:
python [model.py](http://_vscodecontentref_/4)
This will train the model on the dataset and save the trained model as brain_tumor_model.h5 and the training history as history.pkl.

2. Run the Web App
To start the web application, run the app.py script:
streamlit run [app.py](http://_vscodecontentref_/5)

Upload an MRI image, and the app will display the prediction along with a Grad-CAM heatmap.

## Model Dtails
Base Model: MobileNetV2 (pre-trained on ImageNet).
Custom Layers: Added fully connected layers with dropout for binary classification.
Loss Function: Binary Crossentropy.
Optimizer: Adam.

## Dataset
Ensure you have access to a labeled dataset of brain MRI images. You can use publicly available datasets or your own data.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/BrainTumor.git
    ```
2. Navigate to the project directory:
    ```bash
    cd BrainTumor
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Prepare your dataset and place it in the `data/` directory.
2. Run the preprocessing script:
    ```bash
    python preprocess.py
    ```
3. Train the model:
    ```bash
    python train.py
    ```
4. Test the model:
    ```bash
    python test.py
    ```

## Results
After training, the model's performance can be evaluated using the confusion matrix and classification report printed in the console.

## Grad-CAM Visualization
The Grad-CAM heatmap highlights the regions of the MRI image that the model focuses on while making predictions. This is implemented in the make_gradcam_heatmap function in app.py.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
