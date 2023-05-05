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

* 

#### Data Visualization

![image](https://user-images.githubusercontent.com/111785493/236442563-00d4c889-d777-4e0b-8dea-5878ec504d99.png)

### Problem Formulation

* Define:
  * Input / Output
  * Models
    * Describe the different models you tried and why.
  * Loss, Optimizer, other Hyperparameters.

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

*** Comparing RSMLE Error values 
* Linear Regression Model: 1.0762005118946756
* RandomForestRegressor: 0.3389129572301837
* LGMBRegressor: 0.32138335330531576

Seeing Feature Importance per Model
* Linear Regression Model:
![image](https://user-images.githubusercontent.com/111785493/236443009-25d5804a-b0c3-433f-aafb-e4762cba3b65.png)
* RandomForestRegressor:
![image](https://user-images.githubusercontent.com/111785493/236443063-83fd7e18-36a6-43d2-8adc-1cb349ead6ac.png)
* LGMBRegressor:
![image](https://user-images.githubusercontent.com/111785493/236443086-fb5a23fa-8719-428b-812f-0788fc44c95f.png)


### Conclusions



* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

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

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.
* I used the CloudML youtube channel for help with data processing for applying the regression models, as well as the RSMLE error: https://www.youtube.com/watch?v=6HVCuXrsQBs&t=4639s
* https://www.analyticsvidhya.com/blog/2015/06/solution-kaggle-competition-bike-sharing-demand/
* https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset
* 





