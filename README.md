# **TechParva Datathon Project**

Welcome to our TechParva Datathon project repository! Here, we present our submission for the TechParva Datathon competition, where we analyzed data to derive meaningful insights and solutions. Leveraging the power of Python libraries such as Pandas for data manipulation and Scikit-learn for building Random Forest models, we embarked on a journey to extract valuable information from the given datasets.

## Team Members:
- [Bigyan Aryal](https://github.com/bigyan08)
- [Suzan Kharel](https://github.com/Sujan29k)

## Files
    train_data.csv: The training dataset with features and target variable.
    test_data.csv: The test dataset with features for which predictions are to be made.
    predict.py: The script to load data, train the model, evaluate it, and make predictions.
    README.md: This README file.

## Requirements

    Python 3.x
    pandas
    scikit-learn

## Steps to Train, Evaluate, and Predict

### Step 1: Load and Split the Training Data

The training data is split into training and validation sets to evaluate the model's performance.

import pandas as pd
from sklearn.model_selection import train_test_split

### Load the train dataset
train_data = pd.read_csv('path_to_train_data.csv')

### Separate features and target
X = train_data.drop(columns='28')  # Drop the target column
y = train_data['28']

### Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

## Step 2: Train the Model

from sklearn.linear_model import LinearRegression

### Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

## Step 3: Evaluate the Model

Predict the target values for the validation set and calculate the Mean Squared Error (MSE).

from sklearn.metrics import mean_squared_error

### Predict on the validation set
val_predictions = model.predict(X_val)

### Calculate MSE on the validation set
val_mse = mean_squared_error(y_val, val_predictions)
print(f'Validation Mean Squared Error: {val_mse}')

## Step 4: Predict on the Test Dataset and Prepare Submission

Load the test dataset, make predictions, and save the results to a CSV file.

### Load the test dataset
test_data = pd.read_csv('path_to_test_data.csv')

### Assuming the test dataset has an 'index' column and columns '1' to '27' for features
X_test = test_data.drop(columns='index')
index = test_data['index']

### Predict on the test data
test_predictions = model.predict(X_test)

### Combine index and predictions into a single DataFrame
submission_df = pd.DataFrame({
    'index': index,
    'Predictions': test_predictions
})

### Save the combined DataFrame to a CSV file
submission_df.to_csv('path_to_save_predictions.csv', index=False)

Final Note

Ensure the paths to the data files are correctly specified. The script assumes the presence of train_data.csv and test_data.csv in the working directory. Adjust the file paths as needed.

## License

This project is licensed under the MIT License.
