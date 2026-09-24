# 📊 SWYNEX — Data Cleaning & Preparation

A complete **Data Cleaning and Preparation** project built using **Python, Pandas, NumPy, Matplotlib, and Scikit-learn** on the Kaggle **Students Performance in Exams** dataset.

The project demonstrates how raw educational data can be inspected, cleaned, transformed, validated, and prepared for further **Data Analysis, Visualization, and Machine Learning**.

---

## 🚀 Project Overview

Real-world datasets often contain missing values, duplicate records, inconsistent text, invalid values, and outliers.

This project implements a complete data preprocessing pipeline to transform the raw student-performance dataset into:

* A clean, analysis-ready dataset
* An ML-ready dataset

### Workflow

```text
Raw Dataset
     ↓
Data Loading
     ↓
Data Inspection
     ↓
Missing Value Handling
     ↓
Duplicate Removal
     ↓
Data Type Conversion
     ↓
Text Standardization
     ↓
Invalid Value Handling
     ↓
Outlier Detection & Treatment
     ↓
Feature Engineering
     ↓
Categorical Encoding
     ↓
Feature Scaling
     ↓
Data Validation
     ↓
Cleaned / ML-Ready Dataset
```

---

## 📁 Dataset

### Source

**Students Performance in Exams — Kaggle**

🔗 https://www.kaggle.com/spscientist/students-performance-in-exams

### Dataset Information

The dataset contains **1,000 student records and 8 original features**, including:

| Feature                     | Description                              |
| --------------------------- | ---------------------------------------- |
| Gender                      | Student gender                           |
| Race/Ethnicity              | Student race/ethnicity group             |
| Parental Level of Education | Parent's education level                 |
| Lunch                       | Type of lunch program                    |
| Test Preparation Course     | Whether preparation course was completed |
| Math Score                  | Mathematics examination score            |
| Reading Score               | Reading examination score                |
| Writing Score               | Writing examination score                |

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — Data manipulation and analysis
* **NumPy** — Numerical operations
* **Matplotlib** — Data visualization
* **Scikit-learn** — Data preprocessing and scaling

---

## 🧹 Data Cleaning Steps

### 1. Dataset Loading

The raw CSV dataset is loaded using Pandas.

### 2. Data Inspection

The project examines:

* Dataset shape
* Column names
* Data types
* Statistical summaries
* Categorical values

### 3. Missing Value Handling

Missing values are detected automatically.

* Numerical values → Median imputation
* Categorical values → Mode imputation

### 4. Duplicate Removal

Duplicate records are identified and removed.

### 5. Column Name Cleaning

Column names are standardized into a consistent format.

Example:

```text
Math Score
↓
math_score
```

### 6. Text Standardization

Text fields are cleaned by:

* Removing unnecessary spaces
* Standardizing text formatting
* Normalizing known categorical values

### 7. Data Type Conversion

Score columns are converted into appropriate numerical data types.

### 8. Invalid Value Detection

Examination scores are validated against the expected **0–100** range.

### 9. Outlier Detection

The **Interquartile Range (IQR)** method is used to identify potential outliers.

### 10. Outlier Treatment

Extreme values are handled using IQR-based clipping while preserving student records.

---

## ⚙️ Feature Engineering

The project creates additional useful features from the original scores.

### Average Score

```text
Average Score =
(Math + Reading + Writing) / 3
```

### Total Score

```text
Total Score =
Math + Reading + Writing
```

### Performance Category

Students are categorized into:

* Excellent
* Good
* Average
* Poor

### Pass/Fail Status

A student is classified as **Pass** when all three subject scores meet the defined passing threshold.

---

## 🤖 Machine Learning Preparation

For machine-learning workflows, categorical variables are converted using **One-Hot Encoding**.

Numerical features are standardized using:

```python
StandardScaler()
```

This produces a separate ML-ready dataset suitable for downstream machine-learning algorithms.

---

## 📂 Project Structure

```text
SWYNEX-Data-Cleaning-Preparation/
│
├── data.py
│
├── StudentsPerformance.csv
│
├── StudentsPerformance_Cleaned.csv
│
├── StudentsPerformance_ML_Ready.csv
│
└── README.md
```

### File Description

| File                               | Purpose                                         |
| ---------------------------------- | ----------------------------------------------- |
| `data.py`                          | Complete data cleaning and preparation pipeline |
| `StudentsPerformance.csv`          | Original dataset                                |
| `StudentsPerformance_Cleaned.csv`  | Cleaned and analysis-ready dataset              |
| `StudentsPerformance_ML_Ready.csv` | Encoded and standardized dataset                |
| `README.md`                        | Project documentation                           |

---

## 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ritik-09-maker/SWYNEX-Data-Cleaning-Preparation.git
```

### 2. Navigate to the project

```bash
cd SWYNEX-Data-Cleaning-Preparation
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## ▶️ How to Run

Run the Python script:

```bash
python data.py
```

After execution, the project generates the processed datasets:

```text
StudentsPerformance_Cleaned.csv
StudentsPerformance_ML_Ready.csv
```

---

## 📈 Data Validation

The final pipeline validates:

* Missing values
* Duplicate records
* Invalid scores
* Data types
* Dataset dimensions
* Numerical statistics

This ensures that the resulting dataset is suitable for further analysis and modeling.

---

## 📊 Applications

The cleaned dataset can be used for:

* Exploratory Data Analysis (EDA)
* Student performance analysis
* Data visualization
* Statistical analysis
* Machine learning experiments
* Performance prediction
* Educational analytics

---

## 🎯 Learning Outcomes

Through this project, the following practical skills are demonstrated:

* Data collection and loading
* Data quality assessment
* Missing-value handling
* Duplicate detection
* Data standardization
* Outlier detection
* Feature engineering
* Categorical encoding
* Numerical scaling
* Data validation
* Python-based data preprocessing

---

## 🔮 Future Improvements

Possible extensions include:

* Exploratory Data Analysis dashboard
* Correlation analysis
* Interactive visualizations
* Student performance prediction
* Machine-learning model comparison
* Feature importance analysis
* Streamlit-based interactive application

---

## 👨‍💻 Author

### Ritik Gupta

**B.Tech — Computer Science & Engineering (AI & Data Science)**

Interested in:

* Data Science
* Artificial Intelligence
* Machine Learning
* Data Analytics
* Python

---

## ⭐ Acknowledgements

* **Kaggle** — Students Performance in Exams dataset
* **Pandas** — Data manipulation
* **NumPy** — Numerical computing
* **Matplotlib** — Visualization
* **Scikit-learn** — Machine learning preprocessing

---

## 📄 License

This project is intended for **educational and learning purposes**. Please refer to the original dataset source for its applicable dataset terms.

---

### ⭐ If you found this project useful, consider giving the repository a star!
