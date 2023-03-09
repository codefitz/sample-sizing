import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as m
import sklearn.preprocessing as pp

# Load the dataset into a pandas DataFrame.
df = pd.read_csv('data2.csv')

# Impute missing values with the mean of each column
df.fillna(0, inplace=True)

# Get the names of all columns in the DataFrame
all_columns = list(df.columns)

# Select the input and output variable names
#input_column = all_columns[all_columns.index('samples')]
#output_columns = all_columns[:all_columns.index('samples')]

# Split the data into input features (X) and output (y) variables
#X = df[[input_column]]
#y = df[output_columns]

X = df[['WGS']]
y = df[['WES','Storage WGS','Storage WES','TA Cores','TA Memory GB','TA Filesystem GB','DB Cores','DB Memory GB','DB Storage TB','DB Filesystem GB','Web Cores','Web Memory GB','Web Filesystem GB','Infra Cores','Infra Memory GB','Infra Filesystem GB','NFS TB']]

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = ms.train_test_split(X.values, y, test_size=0.2, random_state=42)

# Train a regression model using the training data.
reg = lm.LinearRegression()
reg.fit(X_train, y_train)

# Evaluate the model using the testing data.
y_pred = reg.predict(X_test)
mse = m.mean_squared_error(y_test, y_pred)

# Use the trained model to make a prediction for a new input.
print(df)
wgs_input = input("Enter the number of WGS samples: ")
new_input = [[int(wgs_input)]]
predicted_values = reg.predict(new_input)[0]
print("WGS = ",new_input[0][0])
for i, column in enumerate(y.columns):
    print(f"{column}: {int(predicted_values[i])}")
