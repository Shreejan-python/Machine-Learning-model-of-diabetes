# importing modules
import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# making variable diabetes
diabetes = datasets.load_diabetes()
diabetes_x = diabetes.data[:, np.newaxis, 2]
# traning diabetes
diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-30:]
diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_x_train, diabetes_y_train)
# predicting
diabetes_y_predicted = model.predict(diabetes_x_test)
# printing the mean squared error
print("The mean squared error is ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))
# printing weight and intercepts
print("Weights are : ", model.coef_)
print("Intercept are : ", model.intercept_)

plt.scatter(diabetes_x_test, diabetes_y_test)
plt.plot(diabetes_x_test, diabetes_y_predicted)
plt.show()