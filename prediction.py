'''
This files is developed to predict Earthquakes in the Japanese region.
The file covers the information of the dataset EMAG2V3 dataset, and does pre-processing to find the location of the Earthquake zones, try predicting them pre-meptively.
Through some exploratory data analysis, the insights will be obtained over the regions where the earthquaked have occured.
Further, we will look at some time series analysis of how the variation in the data can help in the prediction.

EMAG2V3 Dataset Metadata: Link to the dataset: https://www.ngdc.noaa.gov/geomag/data/EMAG2/EMAG2_V3_20170530/EMAG2_V3_20170530.zip
Column 1: i ; grid column/longitude index
Column 2: j ; grid row/latitude index
Column 3: LON ; Geographic Longitude WGS84 (decimal degrees)
Column 4: LAT ; Geographic Latitude WGS84 (decimal degrees)
Column 5: SeaLevel ; Magnetic Anomaly Value at Sea Level(nT)
Column 6: UpCont ; Magnetic Anomaly Value at continuous 4km altitude (nT)
Column 7: Code ; Data Source Code (see table below)
Column 8: Error ; Error estimate (nT)

The dataset size is about 4Gb, which is quite difficult to load over the python file. I have added the dataset snippet in the documentation for the project:

For the project, we will be focusing on the features: LAT, LONG, Code, SeaLevel. The code for the Japan is 25. For the specificity of the location, we have filtered out that data.
The field Sealevel has the magnetic anomaly value at the sea level in the measurement of nT, which is one of the important feautre of our implementation. We will focus on these values.
As, natural calamities are unfortunate events, and that time-series data might not be available publicly, we have interpolated some data, so that it helps in prediction, and model training.

From the EMAG2V3 dataset, we have generated another dataset to add the anomalies in the dataset, along with additional column of occurence of Earthquake that we will look over in the codes below.
We will also observe the gradual progression of the data variation. 
'''

#%%
'''
Importing the libraries
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest



#%%
'''Importing the data'''
data = pd.read_csv('japan1.csv')
print(data.head(10))

#%%
'''Data Procesising'''
'''
earthquake_indices = data.index[data['Earthquake'] == 1].tolist()
extended_indices = []

for idx in earthquake_indices:
    start_idx = max(0, idx - 10)  # Ensure we don't go out of bounds
    extended_indices.extend(range(start_idx, idx + 1))

# Remove duplicates and sort indices
extended_indices = sorted(set(extended_indices))

# Create the new dataset with the selected indices
earthquake_data = data.iloc[extended_indices].copy()

# Assign random dates starting from a base date
base_date = datetime(2017, 1, 1)
earthquake_data['Date'] = [base_date + timedelta(days=i) for i in range(len(earthquake_data))]

# Shuffle dates for randomness
np.random.seed(42)  # For reproducibility
earthquake_data['Date'] = np.random.permutation(earthquake_data['Date'].values)

# Sort by Date to maintain chronological order with random starting points
earthquake_data.sort_values(by='Date', inplace=True)

print(earthquake_data.head(50))
'''

#%%
'''EDA-Exploratory Data Analysis'''
data2 = pd.read_csv('earthquake_data_for_arcgis.csv')
data2['Region'] = data2.apply(lambda row: f"{row['Latitude']:.2f}, {row['Longitude']:.2f}", axis=1)

plt.figure(figsize=(10, 8))
sns.scatterplot(data=data2, x='Longitude', y='Latitude', hue='Earthquake', palette='coolwarm')
plt.title('Regions with Earthquake Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Earthquake')
plt.show()

# earthquake_data.to_csv('earthquake_data_for_arcgis.csv', index=False)


#%%
'''Anomaly detection'''
# Select only the relevant columns
features = data2[['Sealevel']]
target = data2['Earthquake']
# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123)
iforest = IsolationForest(contamination=0.1, random_state=42)
iforest.fit(X_train)

# Predict anomalies on the test set
y_pred_if = iforest.predict(X_test)

# Anomalies are labeled as -1, normal points as 1
print("Isolation Forest Predictions:", y_pred_if)
# %%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Map predictions to binary labels (1 for normal, 0 for anomaly)
y_pred_if_binary = [0 if x == -1 else 1 for x in y_pred_if]

# Print accuracy metrics
print("Accuracy:", accuracy_score(y_test, y_pred_if_binary))
print("Classification Report:\n", classification_report(y_test, y_pred_if_binary))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_if_binary))

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_if_binary), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# %%
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Sealevel'], y_pred_if_binary, c=y_pred_if_binary, cmap='coolwarm', label='Predictions')
plt.title('Sealevel vs Anomaly Predictions')
plt.xlabel('Sealevel')
plt.ylabel('Anomaly (0 = Anomaly, 1 = Normal)')
plt.legend()
plt.show()
# %%
plt.figure(figsize=(8, 5))
sns.histplot(data['Sealevel'], kde=True, bins=30)
plt.title('Distribution of Sealevel Values')
plt.xlabel('Sealevel')
plt.ylabel('Frequency')
plt.show()

# %%
# Initialize and fit the LOF model
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred_lof = lof.fit_predict(X_train)

# Convert LOF predictions to binary labels (1 for normal, 0 for anomaly)
y_pred_lof_binary = [0 if x == -1 else 1 for x in y_pred_lof]

# Evaluate LOF on the training set (unsupervised)
print("Local Outlier Factor Results:")
print("Accuracy:", accuracy_score(y_train, y_pred_lof_binary))
print("Classification Report:\n", classification_report(y_train, y_pred_lof_binary))

# %%
# Initialize and fit the Logistic Regression model
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)

# Predict on the test set
y_pred_logreg = logreg.predict(X_test)

# Evaluate Logistic Regression model
print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Classification Report:\n", classification_report(y_test, y_pred_logreg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_logreg), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %%
# Initialize and fit the SVM model
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)

# Predict on the test set
y_pred_svm = svm.predict(X_test)

# Evaluate SVM model
print("Support Vector Machine Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %%
