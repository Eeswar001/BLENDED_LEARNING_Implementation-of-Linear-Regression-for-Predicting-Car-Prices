# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## AlgorithmImport Libraries: Bring in essential libraries such as pandas, numpy, matplotlib, and sklearn.
1. Load Dataset: Import the dataset containing car prices along with relevant features.
2. Data Preprocessing: Manage missing data and select key features for the model, if required.
3. Split Data: Divide the dataset into training and testing subsets.
4. Train Model: Build a linear regression model and train it using the training data.
5. Make Predictions: Apply the model to predict outcomes for the test set.
6. Evaluate Model: Measure the model's performance using metrics like R² score, Mean Absolute Error (MAE), etc.
7. Check Assumptions: Plot residuals to verify assumptions like homoscedasticity, normality, and linearity.
8. Output Results: Present the predictions and evaluation metrics.

## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: Pragatheeswaran K
RegisterNumber:  212225040310

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
df=pd.read_csv('CarPrice_Assignment.csv')
df.head()
x = df[['enginesize','horsepower','citympg','highwaympg']]
y = df['price']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled=scaler.transform(x_test)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("-"*50)

print('Name: PRAGATHEESWARAN K')
print('Reg No:212225040310')
print("-"*50)

print("MODEL COEFFICIENTS:")
for feature,coef in zip(x.columns,model.coef_):
    print(f"{feature:}:{coef:}")
print(f"{'Intercept':}:{model.intercept_:}")
print("-"*50)
print("MODEL PERFORMANCE:")
print(f"{'MSE':}:{mean_squared_error(y_test,y_pred):}")
print(f"{'RMSE':}:{np.sqrt(mean_squared_error(y_test,y_pred)):}")
print(f"{'R-squared':}:{r2_score(y_test,y_pred):}")

# 1. Linearity check
plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Linear Check:Actual vs Prediction Prices")
plt.xlabel("Actual Price($)")
plt.ylabel("Predicted Price($)")
plt.grid(True)
plt.show()

# 2. Independence (Durbin-Watson)
residuals =y_test-y_pred
dw_test =sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson StatisticL {dw_test:.2f}","\n(values close to 2 indicicate no autocorrelation)") 

# 3. Homoscedasticity
plt.figure(figsize=(10,5))
sns.residplot(x=y_pred,y=residuals,lowess=True,line_kws={'color':'red'})
plt.title("Homoscedasticity Check:Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

# 4. Normality of residuals
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals,kde=True,ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()

*/
```

## Output:
<img width="590" height="333" alt="image" src="https://github.com/user-attachments/assets/2eed3206-85c9-451b-a92a-b9258128e23e" />
<img width="1212" height="594" alt="image" src="https://github.com/user-attachments/assets/6225fe3f-9873-4665-9dbb-2e2159e37989" />
<img width="778" height="466" alt="image" src="https://github.com/user-attachments/assets/87121b9e-111b-43a0-af70-f9cc7d50d8d3" />
<img width="805" height="334" alt="image" src="https://github.com/user-attachments/assets/48421b71-9790-498a-98b0-2b95015c95c6" />


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
