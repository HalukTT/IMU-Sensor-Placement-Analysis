# 🏃‍♂️ Investigating IMU Sensor Placement for Human Motion Recognition

**Course:** MYZ307E Project  
**Team Members:** Sehlenur Afşar, Haluk Tamyürek, Buğra Kağan Duman, Yusuf Gültekin  
**Instructor:** Sümeye Nur Karahan  

---

## 📌 Project Overview
Wearable devices demand high energy efficiency and user comfort. Using multiple IMU sensors increases data traffic, synchronization complexity, and overall hardware costs. 

This project investigates the performance trade-off of Human Activity Recognition (HAR) across three different IMU sensor configurations. Our primary goal is to determine if a **single-sensor setup** (Thigh or Lower Back) can capture enough gait dynamics to reach accuracy benchmarks acceptable for resource-constrained embedded systems, compared to a baseline dual-sensor setup.

## 📊 Dataset & Methodology

### 1. The HARTH Dataset
We utilized the **HARTH** (Human Activity Recognition Dataset for Machine Learning) dataset, which contains multi-channel acceleration signals from 31 subjects recorded in free-living environments.
* **Sensor Positions:** Lower Back (Trunk) and Right Lateral Thigh.
* **Sampling Rate:** 50 Hz.

> **⚠️ Note on Hardware Anomaly:** During our literature and repository research, we noted that subjects 6-22 experienced a physical sensor orientation error on the thigh. The dataset providers corrected this using a mathematical rotation matrix. This highlights the importance of robust software preprocessing to compensate for hardware placement variations.

### 2. Feature Engineering Pipeline
Machine learning models require summarized characteristics rather than raw temporal data. Our preprocessing pipeline involves:
* **Windowing:** Raw signals were segmented into **100-sample (2-second) windows**.
* **Statistical Extraction:** For each window, we extracted **5 statistical features** per axis: `Mean`, `Standard Deviation`, `Min`, `Max`, and `Median`.
* **Total Features:** 15 features for single-sensor setups (3 axes * 5 stats), 30 features for the dual-sensor setup.

### 3. Model & Validation
* **Algorithm:** Random Forest Classifier (100 estimators, balanced class weights).
* **Validation Strategy:** Subject-based split (Train/Test) to ensure the model is evaluated on completely unseen individuals, mimicking real-world deployment.

---

## 🏆 Results & Hardware Trade-off

Our comparative experiments yielded the following results across the three configurations:

| Configuration | Accuracy | Weighted F1 | Macro F1 |
| :--- | :---: | :---: | :---: |
| **Lower Back Only** | 62.41% | 60.14% | 32.01% |
| **Thigh Only** | 79.95% | 78.63% | 37.42% |
| **Dual Sensor (Fused)** | **89.33%** | **89.32%** | **49.33%** |

### 💡 Engineering Conclusion
While the **Dual Sensor** setup provides the highest absolute reliability (~89%), it doubles the required bandwidth and energy consumption. 
Placing a single sensor on the **Thigh** provides a high-performance alternative (~80% accuracy), proving to be **17% more informative** than the lower back for gait-dominant activities. For battery-powered, resource-constrained embedded systems, the Thigh-only configuration offers the optimal balance between hardware simplicity and algorithmic accuracy.

---

## 🚀 How to Run Locally

Due to GitHub's file size limits, the large dataset (`.csv` files) is not included in this repository. To run the code and reproduce our results on your local machine, please follow these steps:

### Prerequisites
Make sure you have Python installed along with the required libraries:
`pip install pandas numpy scikit-learn scipy`

### Step-by-Step Guide
1. **Clone the repository:**
   `git clone https://github.com/HalukTT/IMU-Sensor-Placement-Analysis.git`
   `cd IMU-Sensor-Placement-Analysis`
2. **Create the data folder:**
   Create an empty folder named `data` in the root directory of this project.
3. **Download the Dataset:**
   Download the original `.csv` files from the official HARTH repository and place them inside the `data` folder.
4. **Run the Model:**
   Execute the Python script. The script will automatically locate the `data` folder, process the windows, and print the evaluation metrics.
   `python main.py`

---
*Developed for MYZ307E - May 2026*
