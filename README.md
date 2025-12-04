# 🩺 Diabetes Progression Prediction

This project implements an **end-to-end Machine Learning system** for predicting **disease progression after one year** for diabetes patients.  
The project includes:

- Full **Exploratory Data Analysis (EDA)**
- **Data preprocessing** (outliers, scaling...)
- Multiple **regression models** + hyperparameter tuning
- A **Streamlit application** for interactive prediction

This project was developed as part of the **Machine Learning** coursework.

[Streamlit App](https://diabetes-kyo.streamlit.app)

## Dataset

- **Number of samples:** 442  
- **Number of features:** 10  
- **Target variable:** Y (disease progression score)
- **Dataset type:** Regression problem  

[Link to the dataset](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-diabetes?tabs=azureml-opendatasets)

### Features

| Feature | Description |
|--------|-------------|
| **AGE** | Age of the patient |
| **SEX** | 1 = Male, 2 = Female |
| **BMI** | Body Mass Index |
| **BP**  | Mean arterial blood pressure |
| **S1** | Total cholesterol |
| **S2** | LDL cholesterol |
| **S3** | HDL cholesterol |
| **S4** | Cholesterol ratio (TC/HDL) |
| **S5** | Log triglycerides |
| **S6** | Blood glucose level |
| **Y** | Target variable — disease progression after 1 year |

## Results

| Category | Model | MAE | MSE | RMSE | R² |
|----------|--------|------|-----------|-------------|------------|
| Linear | Linear Regression | 42.7941 | 2900.1936 | 53.8534 | 0.4526 |
| Linear | Ridge | 42.8260 | 2887.7536 | 53.7378 | 0.4550 |
| Linear | Lasso | 42.7956 | 2897.6504 | 53.8298 | 0.4531 |
| Linear | ElasticNet | 42.8429 | 2882.9488 | 53.6931 | 0.4559 |
| Trees | Random Forest | 42.7252 | 2774.5614 | 52.6741 | 0.4763 |
| Trees | Gradient Boosting | 42.5822 | 2749.6493 | 52.4371 | 0.4810 |
| Tuned | RF (RandomizedSearchCV) | 43.5732 | 2821.0632 | 53.1137 | 0.4675 |
| Tuned | **XGBoost (Bayesian)** | **41.9275** | **2684.5313** | **51.8125** | **0.4933** |
| Boosting | XGBoost (Default) | 44.8165 | 3217.9541 | 56.7270 | 0.3926 |
| Boosting | CatBoost | 44.3645 | 2872.1155 | 53.5921 | 0.4579 |

## License

This project is for **educational use** only.

## Acknowledgements

- WIUT
- Azure Open Datasets
- Scikit-learn Team  
- Streamlit
