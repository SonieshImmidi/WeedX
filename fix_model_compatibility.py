import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# Create models directory if it doesn't exist
os.makedirs('models_new', exist_ok=True)

# Load Crop Recommendation dataset
print("Loading dataset...")
try:
    # Try to load from standard location
    data = pd.read_csv('Data-processed/Crop_recommendation.csv')
except FileNotFoundError:
    # Check if it's in the notebooks folder
    try:
        data = pd.read_csv('Data-raw/Crop_recommendation.csv')
    except FileNotFoundError:
        # If still not found, download from the source
        print("Dataset not found. Downloading from source...")
        data = pd.read_csv('https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/Crop_recommendation.csv')

print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns")

# Prepare features and target
features = data.drop('label', axis=1)
target = data['label']

# Split the dataset
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=2)

print("Training models with current scikit-learn version...")

# Decision Tree
print("Training Decision Tree...")
DecisionTree = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
DecisionTree.fit(Xtrain, Ytrain)

# Save Decision Tree model
DT_pkl_filename = 'models_new/DecisionTree.pkl'
with open(DT_pkl_filename, 'wb') as DT_Model_pkl:
    pickle.dump(DecisionTree, DT_Model_pkl)
print(f"Saved {DT_pkl_filename}")

# Naive Bayes
print("Training Naive Bayes...")
NaiveBayes = GaussianNB()
NaiveBayes.fit(Xtrain, Ytrain)

# Save Naive Bayes model
NB_pkl_filename = 'models_new/NBClassifier.pkl'
with open(NB_pkl_filename, 'wb') as NB_Model_pkl:
    pickle.dump(NaiveBayes, NB_Model_pkl)
print(f"Saved {NB_pkl_filename}")

# SVM
print("Training SVM...")
# Normalize data for SVM
norm = MinMaxScaler().fit(Xtrain)
X_train_norm = norm.transform(Xtrain)
X_test_norm = norm.transform(Xtest)

SVM = SVC(kernel='poly', degree=3, C=1)
SVM.fit(X_train_norm, Ytrain)

# Save SVM model
SVM_pkl_filename = 'models_new/SVMClassifier.pkl'
with open(SVM_pkl_filename, 'wb') as SVM_Model_pkl:
    pickle.dump(SVM, SVM_Model_pkl)
print(f"Saved {SVM_pkl_filename}")

# Random Forest
print("Training Random Forest...")
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain, Ytrain)

# Save Random Forest model
RF_pkl_filename = 'models_new/RandomForest.pkl'
with open(RF_pkl_filename, 'wb') as RF_Model_pkl:
    pickle.dump(RF, RF_Model_pkl)
print(f"Saved {RF_pkl_filename}")

print("\nAll models have been retrained and saved with the current scikit-learn version.")
print("To use these models, copy them from the 'models_new' folder to the 'models' folder.") 