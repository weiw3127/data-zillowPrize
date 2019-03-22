# Zillow Prize: Zillow’s Home Value Prediction
Machine Learning Solution for [Zillow Prize: Zillow’s Home Value Prediction (Zestimate)](https://www.kaggle.com/c/zillow-prize-1)


The contest was hosted by [Zillow](https://www.zillow.com/), the leading real estate marketplace. 
“Zestimates” are estimated home values based on 7.5 million statistical and machine learning models that analyze hundreds of data points on each property. Zillow Prize, a competition with a one million dollar grand prize, is challenging the data science community to help push the accuracy of the Zestimate even further. Winning algorithms stand to impact the home values of 110M homes across the U.S.  

This repository contains two files: 
1. **feature.py:** 
  feature engineering process that generates the most relevant data for the machine learning model. 
2. **model.py:** machine learning pipeline that produces the prediction
	- Main model: XGB, SVR, Gradient Boost, LightGBM, Keras, Ridge Linear.
	-  Ensemble method: XGB.

The result is evaluated on Mean Absolute Error between the predicted log error and the actual log error. The Public LB score is **0.07507 (154th of 3779, top 5%)**.

# Instruction
*	download data from the [original dataset](https://www.kaggle.com/c/zillow-prize-1/data)
* run python `./feature.py` to get the features 
*	run python `./model.py` to generate the best predicting result.
