# Diabetes Progression Prediction

This app predicts **disease progression after one year** for diabetes patients.

This project was developed as part of the **Machine Learning** coursework.

[Streamlit App](https://diabetes-kyo.streamlit.app)

## Dataset

- **Number of samples:** 442  
- **Number of features:** 10  
- **Target variable:** Y (disease progression score)
- **Dataset type:** Regression problem  
- [Link to the dataset](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-diabetes?tabs=azureml-opendatasets)

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

| Model                                                | MAE        | MSE          | RMSE       | R²        |
|------------------------------------------------------|------------|--------------|------------|-----------|
| Random Forest (RandomizedSearch)                     | 43.573224  | 2821.063230  | 53.113682 | 0.467538  |
| Random Forest (RandomizedSearch) without outliers    | 46.805187  | 3267.544179  | 57.162437 | 0.504618  |
| XGBoost (Bayesian)                                   | 41.927536  | 2684.531250  | 51.812462 | 0.493308  |
| XGBoost (Bayesian) without outliers                  | 46.250881  | 3212.671631  | 56.680434 | 0.512937  |
| CatBoost                                             | 44.364513  | 2872.115500  | 53.592122 | 0.457902  |
| CatBoost without outliers                            | 47.263629  | 3306.880608  | 57.505483 | 0.498654  |
| XGBoost                                              | 44.816467  | 3217.954102  | 56.727014 | 0.392627  |
| XGBoost without outliers                             | 48.050686  | 3479.955078  | 58.991144 | 0.472415  |
| Ridge                                                | 42.826022  | 2887.753622  | 53.737823 | 0.454951  |
| Ridge without outliers                               | 44.939837  | 3168.329914  | 56.287920 | 0.519660  |
| Lasso                                                | 42.795632  | 2897.650390  | 53.829828 | 0.453083  |
| Lasso without outliers                               | 45.064143  | 3174.742783  | 56.344856 | 0.518687  |
| ElasticNet                                           | 42.842929  | 2882.948793  | 53.693098 | 0.455858  |
| **ElasticNet without outliers**                      | **44.866164** | **3165.361387**  | **56.261544** | **0.520110**  |
| Linear Regression without outliers                   | 45.073266  | 3175.043351  | 56.347523 | 0.518642  |
| Random Forest without outliers                       | 47.081602  | 3414.595935  | 58.434544 | 0.482324  |
| Gradient Boosting without outliers                   | 47.606703  | 3351.964218  | 57.896150 | 0.491819  |
| Linear Regression                                    | 42.794095  | 2900.193628  | 53.853446 | 0.452603  |
| Random Forest                                        | 42.725168  | 2774.561442  | 52.674106 | 0.476315  |
| Gradient Boosting                                    | 42.582154  | 2749.649262  | 52.437098 | 0.481017  |

## Usage

Clone the repository
```
git clone git@github.com:kyojin22/diabetes.git
```

Create virtual environment and install required libraries

macOS / Linux
```
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

Windows
```
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt
```

Run streamlit app
```
streamlit run streamlit_app/app.py
```

## License

Licensed under the [MIT License](/LICENSE)
