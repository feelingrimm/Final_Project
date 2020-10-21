#%matplotlib inline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
import os
# read the data
df = pd.read_csv("data.csv")
df.head(10)
# Dropping the id columns
cleaned_df = df[["diagnosis", "texture_mean", "perimeter_mean", "smoothness_mean", "symmetry_mean", "fractal_dimension_mean", "texture_se", "smoothness_se", "compactness_se", "symmetry_se"]]
# Setting up the machine
y = cleaned_df["diagnosis"]
X = cleaned_df.drop(columns=["diagnosis"])
# Set features. This will also be used as your x values.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = MinMaxScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
# Step 1: Label-encode data set
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
encoded_y_train = label_encoder.transform(y_train)
encoded_y_test = label_encoder.transform(y_test)
# Step 2: Convert encoded labels to one-hot-encoding
y_train_categorical = to_categorical(encoded_y_train)
y_test_categorical = to_categorical(encoded_y_test)

# Create a Deep Learning Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Create model and add layers
model = Sequential()
model.add(Dense(units=10, activation='relu', input_dim=9))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=2, activation='softmax'))
# Compile and fit the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    X_train_scaled,
    y_train_categorical,
    epochs=100,
    shuffle=True,
    verbose=2
)
model_loss, model_accuracy = model.evaluate(
    X_test_scaled, y_test_categorical, verbose=2)
print(
    f"Normal Neural Network - Loss: {model_loss}, Accuracy: {model_accuracy}")
model.save("cancer_data_model.h5")
import joblib
joblib.dump(scaler,"cancer_data.pkl")