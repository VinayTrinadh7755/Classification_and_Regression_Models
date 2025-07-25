# Classification_and_Regression_Models

A hands-on Python project building data preprocessing pipelines, logistic regression, OLS linear & ridge regression, and elastic net—all implemented without scikit-learn or other ML libraries.

## 🔍 Project Overview
This assignment is structured in five parts using real-world “noisy” datasets:

• Part I – Data Analysis & Preprocessing  
• Part II – Logistic Regression via Gradient Descent  
• Part III – Ordinary Least Squares (OLS) Linear & Ridge Regression  
• Part IV – Elastic Net Regression via Gradient Descent  
• Part V - Applying ML methods to a Buffalo-related dataset & boosting penguin accuracy

Each part is delivered as a self-contained Jupyter notebook, complete with plots, pickled weights, and preprocessed CSVs.

## 🎯 Why This Matters
Recruiters and teams value engineers who:  
- Understand data cleaning, imputation, outlier handling, encoding, and normalization  
- Implement core ML algorithms (sigmoid, gradient descent, closed-form OLS) from ground up  
- Tune hyperparameters and regularization (L1, L2, Elastic Net) for robust models  
- Visualize results and interpret metrics (accuracy, MSE, loss curves)  
- Package reproducible code with notebooks, reports, and saved artifacts  

## 🗂 Table of Contents
1. Quick Start  
2. Notebooks Reference  
3. Architecture & Design  
4. Key Features  
5. Project Structure  
6. Usage Examples  
7. Tech Stack  
8. Future Enhancements  
9. Contributing  
10. License  
11. Authors & Contact  

---

## 🔧 Quick Start
```bash
# 1. Clone repo  
git clone https://github.com/vinaytrinah7755/Classification_and_Regression_Methods.git  
cd Classification_and_Regression_Methods

# 2. Create virtual environment & install deps  
python3 -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt  

# 3. Launch Jupyter  
jupyter notebook  
```

Open and run the notebooks in order:
- penguins_part1.ipynb
- penguins_part2.ipynb
- diamonds_part1.ipynb
- diamonds_part4.ipynb
- emissions_by_country_part1.ipynb
- emissions_by_country_part3.ipynb
- buffalo_covid_part1.ipynb
- buffalo_covid_part2.ipynb

## 📜 Notebooks Reference

| Notebook | Purpose | Outputs |
|----------|---------|---------|
| *_part1.ipynb | Data cleaning, visualization, feature engineering | *_preprocessed.csv, plots |
| *_part2.ipynb | Binary logistic regression via gradient descent | *_part2_weights.pickle, accuracy |
| *_part3.ipynb | OLS linear regression & ridge regression | *_part3_weights.pickle, MSE |
| *_part4.ipynb | Elastic Net regression via gradient descent | *_part4_weights.pickle, loss |
| bonus.ipynb (optional) | Buffalo dataset modeling & penguin accuracy tuning | bonus weights, results |

## 🏗 Architecture & Design

- Modular preprocessing functions: handle missing values, string normalization, outlier IQR, encoding, normalization  
- LogisticRegressionGD class: implements sigmoid, cost, gradient descent, fit, predict  
- Closed-form solvers: OLS and ridge using matrix algebra  
- ElasticNetGD class: gradient descent minimizing combined L1/L2 penalty  
- Consistent workflow: load → preprocess → split → train → evaluate → save weights

```
┌──────────┐    preprocess    ┌──────────┐    train      ┌──────────┐  
│ raw data ├────────────────▶│ cleaned  ├────────────▶│ models   │  
└──────────┘                  └──────────┘              └──────────┘  
```

## 🔑 Key Features
**Part I – Data Analysis & Preprocessing**  
- Handle missing data (drop/impute), mismatched strings, outliers  
- Encode categorical features (one-hot, label), normalize numerical  
- Generate 5+ insightful visualizations per dataset

**Part II – Logistic Regression**  
- Train on penguin data, tune learning rate & iterations  
- Plot loss curves, achieve >64% accuracy; best ≈89.9%

**Part III – Linear & Ridge Regression**  
- Solve OLS with closed-form, compare against ridge (L2)  
- Report train/test MSE; ridge reduces overfitting

**Part IV – Elastic Net Regression**  
- Combine L1 & L2 penalties, test zero/random/Xavier init  
- Implement early stopping, compare convergence behaviors

**Bonus Tasks**  
- Apply methods to a Buffalo Open Data dataset (>1k entries)  
- Push penguin classifier beyond 85% with advanced techniques

## 📁 Project Structure
```
Classification_and_Regression_Methods/
├── datasets/  
│   ├── penguins.csv  
│   ├── diamond.csv 
│   ├── emissions_by_country.csv
│   ├── buffalo_covid.csv

├── notebooks/
│   ├── penguins
│     ├── penguins_part1.ipynb  
│     ├── penguins_part2.ipynb   
│   ├── emissons_by_country
│     ├── emissons_by_country_part1.ipynb   
│     ├── emissons_by_country_part3.ipynb   
│   ├── diamonds
│     ├── diamonds_part1.ipynb   
│     ├── diamonds_part4.ipynb  
│   ├── buffalo_covid
│     ├── buffalo_covid_part1.ipynb  
│     ├── buffalo_covid_part2.ipynb    
├── outputs/  
│   ├── *.csv           # preprocessed datasets  
│   ├── *.pickle        # saved model weights  
├── requirements.txt  
├── LICENSE  
└── README.md  
```

## 🚀 Usage Examples

```python
# In Part II notebook cell:
from logistic import LogisticRegressionGD
model = LogisticRegressionGD(lr=0.004, n_iters=200_000)
model.fit(X_train, y_train)
print("Test Accuracy:", model.score(X_test, y_test))  
# ⇒ ~0.8986
```

```python
# In Part III notebook cell:
from linear import OLS, Ridge
w_ols = OLS().fit(X_train, y_train)
mse_test = OLS().mse(X_test, y_test)
```

## 🛠 Tech Stack
- Language: Python 3.8+  
- Core: NumPy, Pandas  
- Visualization: Matplotlib, Seaborn  
- Notebook: Jupyter  
- Pickle: Model weights serialization  

## 🌱 Future Enhancements
- Integrate scikit-learn pipelines for benchmarking  
- Add parallelized/mini-batch gradient descent  
- Hyperparameter search (Grid/Random/Bayesian)  
- Dashboard for real-time metric visualization  

## 🤝 Contributing
1. Fork this repository  
2. Create a feature branch (`git checkout -b feature/xyz`)  
3. Commit your changes (`git commit -m "Add xyz"`)  
4. Push and open a PR for review  

## 📜 License
This project is licensed under the MIT License. See LICENSE for details.

## 👥 Authors & Contact
**Vinay Trinadh Naraharisetty**    
[GitHub](https://github.com/VinayTrinadh7755)  
[LinkedIn](www.linkedin.com/in/vinay-trinadh-naraharisetty)
  
Thank you for exploring our ML implementations!
