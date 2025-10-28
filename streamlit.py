import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Classification import Perceptron, Adaline, confusion_matrix_manual, split_train_test_per_class

 
st.title("🐧 Perceptron & Adaline Classifier for Penguins")

# Load data
penguins = pd.read_csv("Preprocessed.csv")

 
feature1 = st.selectbox("Select Feature 1", penguins.columns[1:8])
feature2 = st.selectbox("Select Feature 2", penguins.columns[1:8])

class_options = ["Adelie", "Chinstrap", "Gentoo"]
class1_label = st.selectbox("Select Class 1", class_options)
class2_label = st.selectbox("Select Class 2", [c for c in class_options if c != class1_label])

species_mapping = {"Adelie": 0, "Gentoo": 1, "Chinstrap": 2}
class1 = species_mapping[class1_label]
class2 = species_mapping[class2_label]

learning_rate = st.number_input("Learning Rate (η)", value=0.001, step=0.001, format="%.6f")
epochs = st.number_input("Number of Epochs", value=100, step=1)
add_bias = st.checkbox("Add Bias", True)
algorithm = st.radio("Choose Algorithm", ["Perceptron", "Adaline"])
mse_threshold = st.number_input("MSE Threshold (Adaline only)", value=0.0001, step=0.0001, format="%.6f")

 
if st.button("Train Model"):
    subset = penguins[penguins["Species"].isin([class1, class2])].copy()
    subset["Label"] = np.where(subset["Species"] == class1, 1, -1)
    X = subset[[feature1, feature2]].values
    y = subset["Label"].values

    X_train, y_train, X_test, y_test = split_train_test_per_class(X, y, 30, 20)

     
    if algorithm == "Perceptron":
        model = Perceptron(input_size=2, learning_rate=learning_rate, epochs=epochs, add_bias=add_bias)
        model.train(X_train, y_train)
    else:
        model = Adaline(eta=learning_rate, epochs=epochs, mse_threshold=mse_threshold, add_bias=add_bias)
        model.fits(X_train, y_train)

    y_pred = model.predict(X_test)
    cm, acc = confusion_matrix_manual(y_test, y_pred)

     
    st.subheader("Confusion Matrix (Manual)")
    st.write(pd.DataFrame(cm, columns=["Pred +1", "Pred -1"], index=["True +1", "True -1"]))
    st.write(f"**Accuracy:** {acc * 100:.2f}%")

     
    st.subheader("Decision Boundary Visualization")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label=class1_label)
    ax.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='red', label=class2_label)

    x_vals = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
    y_vals = model.get_boundary(x_vals)
    ax.plot(x_vals, y_vals, 'g-', label='Decision Boundary')

    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
