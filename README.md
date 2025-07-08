#  Autoencoder-Based Anomaly Detection in Time-Series Data

A deep learning pipeline for unsupervised anomaly detection using autoencoders, designed and evaluated on real-world time-series datasets. This project demonstrates modeling, thresholding, and visualization strategies to identify deviations from learned normal patterns.

---

##  Project Summary

- Objective: Detect anomalies by modeling baseline behavior using deep autoencoders  
- Achieved **>80% test accuracy** with 3 distinct architectures  
- Evaluated on reconstruction error distribution and anomaly thresholding  
- Work emphasizes temporal patterns, regularization, and evaluation analysis  

---

##  Architectural Design

Three distinct autoencoder variants were designed and compared:

| Model Type            | Key Features |
|-----------------------|--------------|
| **Dense Autoencoder** | Fully connected layers, ReLU + Sigmoid activations |
| **LSTM Autoencoder**  | Temporal modeling, sequence-to-sequence design |
| **Conv1D Autoencoder**| Pattern recognition in sequential data, receptive fields |

Each model follows the compression-decompression structure with a bottleneck layer and uses **MSE loss** as the objective. All models incorporate **L2 weight regularization** and dropout to mitigate overfitting.

---

##  Dataset Summary

- Time-series dataset selected from open benchmarks (e.g., Yahoo S5, NAB, Backblaze)  
- Focus on **anomalous trend detection**: spikes, dips, or deviations  
- Normalization applied per feature; missing values handled through interpolation and imputation  
- Split ratio: **Train / Validation / Test = 70 / 15 / 15**

---

##  Evaluation Methodology

- **Loss Function**: Mean Squared Error (MSE)  
- **Optimization**: Adam optimizer with learning rate tuning  
- **Regularization**: L2 norm penalties and dropout  
- **Thresholding**: Statistical analysis on reconstruction error distribution  
- **Metrics**:  
  - Accuracy, F1-score  
  - Mean Absolute Error (MAE), Root Mean Square Error (RMSE)  
  - Precision, Recall for anomaly detection  

---

##  Visual Results

This section showcases the architectural design, training curves, and final evaluation of multiple autoencoder-based models developed for time-series anomaly detection.

---

###  Base Model Architectures

#### 🔷 Model 1: 1D Convolutional Autoencoder

A Convolutional Autoencoder using `nn.Conv1d` and `nn.MaxPool1d` was implemented to capture localized patterns in time-series sequences.

- **Activation Function**: ReLU  
- **Loss Function**: Mean Squared Error (MSE)  
- **Optimizer**: Adam  

<img src="https://github.com/user-attachments/assets/e8fe7488-b911-40de-8b95-eb2823372ac8" width="700"/>

---

#### 🔷 Model 2: LSTM Autoencoder

An LSTM-based autoencoder designed to handle temporal dependencies in sequential data.

- **Encoder**: LSTM (32 hidden units)  
- **Decoder**: LSTM  
- **Loss Function**: MSE  
- **Optimizer**: Adam  

<img src="https://github.com/user-attachments/assets/0384187a-a878-4a6f-b9e9-fe79ee087043" width="700"/>

---

#### 🔷 Model 3: Dense Autoencoder

A fully connected architecture to learn global representations from flattened input.

- **Encoder**: Dense layers (Input → 64 → 32)  
- **Decoder**: Dense layers (32 → 64 → Output)  
- **Activation**: ReLU  
- **Loss Function**: MSE  
- **Optimizer**: Adam  

<img src="https://github.com/user-attachments/assets/b29ef8ea-b36f-4f2c-833f-c91b00bfff9f" width="680"/>

---

###  Model Training Curves

####  Base Model Loss Comparison

<img src="https://github.com/user-attachments/assets/4938d8ce-f613-4be2-9471-721fbdc72eba" width="1000"/>

---

####  Version 2 Model Loss Comparison

<img src="https://github.com/user-attachments/assets/47340f4b-9e3b-4321-b84c-9e4e12e5da36" width="1000"/>

---

####  Results Table – Six Models

<img src="https://github.com/user-attachments/assets/6227df5f-7d6f-4edf-8f21-9935f20af724" width="300"/>

---

####  Version 3 Model Loss Comparison

<img src="https://github.com/user-attachments/assets/7dcab751-6776-4984-9616-c79d5b4dc885" width="1000"/>

---

####  Results Table – Nine Models

<img src="https://github.com/user-attachments/assets/a151c595-0879-4932-b683-a330edf98703" width="300"/>

---

###  Best Model Selection

After rigorous evaluation across nine variants of three base architectures, the **Conv1D Autoencoder V2** was selected as the best-performing model.

**Why Conv1D V2?**
- Efficient and fast to train  
- Low MSE: `0.000025`  
- Low Std Dev: `0.000006`  
- Robust performance with no overfitting  
- Best trade-off between generalization and interpretability  

<img src="https://github.com/user-attachments/assets/b17a5642-ab86-44c3-9db0-fa80d6c40b54" width="850"/>

---

####  Final Evaluation of Best Model

<img src="https://github.com/user-attachments/assets/7d141a12-90c5-458a-ab01-c1c79eca4870" width="310"/>

---

####  Best Model Loss Curve After Tuning

<img src="https://github.com/user-attachments/assets/af05365c-9979-4ecd-8f66-c79079d3b7aa" width="860"/>

---

###  Reconstruction Analysis

####  Distribution of Reconstruction Errors (Test Set)

<img src="https://github.com/user-attachments/assets/b6ea93e7-a655-4744-aa80-64fd7fb0c2e2" width="890"/>

---

###  Anomaly Detection Visualization

#### Conv1D Autoencoder — Anomaly Detection Output

<img src="https://github.com/user-attachments/assets/b4679b9b-a94b-41e3-a7e5-df3a3dcd8ee6" width="1300"/>

---

##  Key Findings

- **Dense autoencoder** provided fast convergence but lacked temporal nuance  
- **LSTM model** excelled on sequential features, yielding higher anomaly separation  
- **Conv1D** found subtle local patterns, but performance was dataset-dependent  
- **Thresholding via reconstruction error distribution** was effective and interpretable  
- **Hyperparameter tuning** (dropout rate, hidden size, batch size) had significant impact  

---
