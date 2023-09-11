import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset into a Pandas DataFrame
data = pd.read_csv('F:\SGP\diabetes-dataset.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Diabetic', axis=1), data['Diabetic'], test_size=0.1, random_state=42)

# Create a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=42)

# Fit the model on the training data
rf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy * 100))

