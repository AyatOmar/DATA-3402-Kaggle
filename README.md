![](UTA-DataScience-Logo.png)

# Bike Sharing Demand Regression Models

* This repository is aimed to predict the hourly bike rental demand for a bike-sharing service in a city based on historical data of bike rental usage patterns, weather data, and other factors. The goal is to build 3 regression models that accurately predict bike rental demands.
https://www.kaggle.com/c/bike-sharing-demand


## Overview

* In the Kaggle Bike Sharing Demand challenge, the three machine learning algorithms (Linear Regression, RandomForestRegressor, LGBMRegressor) can be used to predict the total count of bikes rented in a given hour, based on various input features such as the weather, time of day, day of week, etc. 
* The Kaggle Bike Sharing Demand challenge provides a training dataset with various input features and the corresponding bike rental counts, and a testing dataset without the bike rental counts. The goal is to use the training dataset to build a regression model, and then use the model to predict the bike rental counts in the testing dataset. The submissions are evaluated based on the RMSLE (Root Mean Squared Logarithmic Error) between the predicted and true values.

## Summary of Workdone

* Data processing was done to convert the datetime column to each element (year, day, hour, etc..). After visualizing the count feature, skewness was found, taking the natural log of the 'count' variable, the values were transformed to a new scale that was more suitable for analysis. Root Mean Squared Logarithmic Error (RSMLE) function was applied to evaluate the performance of the regression models. Three popular machine learning algoritms were used: Linear Regression, RandomForestRegressor, LGMBRegressor to predict the total count of bikes rented in a given hour.

### Data

* Data:
  * Data Fields
    * datetime - hourly date + timestamp  
    * season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
    * holiday - whether the day is considered a holiday
    * workingday - whether the day is neither a weekend nor holiday
    * weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
    2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
    4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
    * temp - temperature in Celsius
    * atemp - "feels like" temperature in Celsius
    * humidity - relative humidity
    * windspeed - wind speed
    * casual - number of non-registered user rentals initiated
    * registered - number of registered user rentals initiated    
  * Size: 1.12 MB
  * Instances: A training dataset with 10886 instances (rows) and 12 variables (columns).A testing dataset with 6493 instances (rows) and 9 variables (columns). The target variable "count" is not included in the testing dataset and is instead used by Kaggle to evaluate the performance of the submitted models.

#### Preprocessing / Clean up

* before data processing
![image](https://user-images.githubusercontent.com/111785493/236451979-800cbc87-0cb5-4c86-a6d0-b4d96676622d.png)
* after data processing
![image](https://user-images.githubusercontent.com/111785493/236452028-32c25d45-4f5c-448d-8c25-e66a403f4485.png)

* By taking the natural logarithm of the 'count' variable, the values are transformed to a new scale that is more suitable for analysis, particularly when dealing with data that exhibits a wide range of values. The transformation is a common technique in data analysis, especially when the response variable is count data, to improve model fit and to reduce the effect of outliers.

#### Data Visualization

![image](https://user-images.githubusercontent.com/111785493/236442563-00d4c889-d777-4e0b-8dea-5878ec504d99.png)

### Problem Formulation
* Linear Regression: Linear regression is a simple, yet effective model for regression problems, especially when the relationship between the input variables and the output variable is linear. In this case, linear regression can capture the linear relationship between the bike rental demand and the input variables, such as temperature, humidity, and windspeed.

* Random Forest Regressor: Random Forest Regressor is an ensemble learning algorithm that can handle non-linear relationships between the input variables and the output variable. It builds multiple decision trees on different subsets of the input data and combines their predictions to make a final prediction. In this case, Random Forest Regressor can capture the non-linear relationships between the bike rental demand and the input variables, which may not be captured by a linear model.

* LightGBM Regressor: LightGBM Regressor is a gradient boosting algorithm that uses histogram-based binning and other optimizations to improve training speed and memory usage. It is a powerful algorithm that can handle large datasets and complex models. In this case, LightGBM Regressor can handle the large number of input variables and their interactions, which may not be captured by simpler models such as linear regression.
 
* Using a combination of linear and non-linear models such as Linear Regression, Random Forest Regressor, and LightGBM Regressor can help capture the various relationships between the input variables and the output variable and improve the accuracy of the predictions.

### Performance Comparison

*** Comparing RSMLE Error values 
* Linear Regression Model: 1.0762005118946756
* RandomForestRegressor: 0.3389129572301837
* LGMBRegressor: 0.32138335330531576

### Conclusions




### Future Work

* There are so many more regression models that can be tried and compared, If I had more time, I would definitely try other machine learning algorithms such as Gradient Boosting Regressor, Support Vector Regressor, XGBoost Regressor.

## How to reproduce results

* Reproducing the training on different models is very simple, in the code there is a function called 'evaluate' that takes a machine learning model input and optional paramaters (name and values). The function prints the name of the model class and its RMSLE score on the test dataset, and returns the trained model object and its predictions on the test datase

### Overview of files in repository

* attempt 2.ipynb: this is my final draft and where my final submission lies
* Bike Sharing Demand.ipynb: this was the first attempt, trying different visualization techniques 

### Software Setup
* numpy
* pandas
* seaborn
* matplotlib
* lightgbm
* sklearn

### Data
* The dataset can be found on the Kaggle website
https://www.kaggle.com/competitions/bike-sharing-demand/data

### Training
* The training for each model took a few seconds. 
* Load the training dataset and split it into features and target variables (X_train and y_train, respectively).
* Preprocess the data as necessary (e.g. scale the numerical features, encode categorical variables).
* Instantiate the chosen model class with any desired hyperparameters.
* Train the model using the fit method, passing in the training features (X_train) and targets (y_train).
* Once the model is trained, use it to make predictions on the test dataset (X_test).
* Evaluate the model's performance using the RMSLE metric by comparing the predicted values to the true target values in the test dataset (y_test).

#### Performance Evaluation



## Citations

* Provide any references.
* I used the CloudML youtube channel for help with data processing for applying the regression models, as well as the RSMLE error: https://www.youtube.com/watch?v=6HVCuXrsQBs&t=4639s
* https://www.analyticsvidhya.com/blog/2015/06/solution-kaggle-competition-bike-sharing-demand/
* https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset
* 





