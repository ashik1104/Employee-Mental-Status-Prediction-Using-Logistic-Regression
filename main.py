import pandas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pandas.read_csv('HR_comma_sep.csv')
# print(data.shape)
#  step-1: missing data for any row any column
# print(data.isnull().values.any())

# step-2: check data types
# print(data.dtypes)

# step-3: check the unique values and replace those with numbers
# print(data.salary.unique())
# print(data.Department.unique())
replace_values = {'salary': {'low': 1, 'medium': 2, 'high': 3}}
data.replace(replace_values, inplace=True)
# print(data)

# step-4: get dummies for the department
dummies = pd.get_dummies(data.Department)
# print(dummies)

#  step-5: merge dummies (dummies columns) with the original data
merged = pd.concat([data, dummies], axis='columns')
# print(merged)

# step-6: Drop unnecessary data
final_data = merged.drop(['Department', 'technical'], axis='columns')
# print(list(final_data.columns))

# step-7: plotting data
# plt.scatter(x=final_data.salary, y=final_data.left)
# plt.scatter(x=final_data.satisfaction_level, y=final_data.left)
plt.scatter(x=final_data.time_spend_company, y=final_data.left)
# plt.show()

# step-8: model train and test and check accuracy
x = final_data.drop('left', axis='columns')
y = final_data.left
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
model = LogisticRegression()
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print('Accuracy :', accuracy)

# step-9: test with a customized value
result = model.predict([[0.85, 0.87, 6, 232, 5, 0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
print('Result :', result)
