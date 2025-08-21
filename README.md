# 🔬 Breast Cancer Classification Project

This project implements a **breast cancer classification model** using **Decision Tree** 🌳 and **Logistic Regression** 📊 classifiers.  
It predicts whether a tumor is **malignant** or **benign** based on key medical features extracted from cell nuclei.  
A **Streamlit web app** ⚡ is developed to allow real-time predictions through user inputs, providing a practical diagnostic support tool.  

🔗 **Live App:** [Breast Cancer Classification Web App](https://breastcancerclassificationmodel-85.streamlit.app/)

---

## 📂 Dataset

- **Source:** [Breast Cancer Wisconsin Diagnostic Dataset on Kaggle 🗂️](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)  
- Contains **569 samples** and **32 columns** (30 numerical features, 1 ID column, and 1 target column `diagnosis`).  
- The target variable `diagnosis` indicates tumor status: **malignant (M)** or **benign (B)**.  
- No missing values ✅ or duplicate rows 🔍.  

---

## 🧹 Data Preprocessing

- Converted `diagnosis` to numeric labels: **M → 1**, **B → 0**.  
- Dropped `id` column as it holds no predictive value.  
- Checked for and confirmed no missing values or duplicates.  
- Detected and removed outliers using boxplots 📉.  
- Normalized features using **MinMaxScaler** 📏.  

---

## 📊 Exploratory Data Analysis (EDA) & Visualization

- Visualized feature distributions using histograms.  
- Displayed diagnosis class distribution with a pie chart 🥧 (Benign: 63.7%, Malignant: 37.3%).  
- Created **correlation heatmap** to identify strongest feature-target relations.  
- Used bar plots to compare feature values by diagnosis.  
- Boxplots helped detect and remove outliers.  

---

## 🎯 Feature Selection

- Removed features with correlation below ±0.5 to reduce complexity.  
- Used **Decision Tree Classifier** 🌳 for feature importance.  
- Selected 7 key features:

concavity_worst
radius_mean
concave_points_mean
area_se
radius_worst
perimeter_worst
concave_points_worst

- These features were scaled and used for modeling.  

---

## 🚀 Models & Performance

| Model               | Features Used                                    | Performance on Test Set                     |
|---------------------|-------------------------------------------------|---------------------------------------------|
| Logistic Regression 📊 | Single feature: `concave_points_worst`          | Accuracy: **91%** <br> Precision: **92%** <br> Recall: **84%** <br> F1 Score: **88%** |
| Decision Tree 🌳      | Selected 7 features with max depth = 3           | Accuracy: **94%** <br> Precision: **95%** <br> Recall: **88%** <br> F1 Score: **92%** |

---

## 🌐 Deployment

- Saved trained **Decision Tree** model and scaler using joblib 💾.  
- Integrated models into a **Streamlit web app** ⚡ for real-time tumor classification.  

---

## 📄 Project Report

For a detailed explanation of the analysis, preprocessing, modeling, and results, see the [Full Project presentation](project_presentation.pdf).  

---

💡 **Note:**  
The `diagnosis` target variable is categorical, encoded as **1 for malignant** and **0 for benign tumors**.