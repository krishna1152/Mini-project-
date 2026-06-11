from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Binarizer
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier  
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

global model_folder
model_folder = 'models'
os.makedirs(model_folder,exist_ok=True)

accuracy=[]
precision=[]
recall=[]
fscore=[]




def uploadDataset():
    global filename, categories
    filename = filedialog.askdirectory(initialdir="Dataset")
    if not filename:
        return

    text.delete('1.0', END)
    text.insert(END, f"Folder Loaded:\n{filename}\n\n")

    categories = [
        d for d in os.listdir(filename)
        if os.path.isdir(os.path.join(filename, d))
    ]

    text.insert(END, "Subfolders found:\n")
    for label in categories:
        text.insert(END, f"- {label}\n")


def extract_features(file_path):
    global y, sr, features, mfcc, mfcc_mean
    y, sr = librosa.load(file_path, sr=None)  # Load audio file
    
    # Feature extraction
    features = []
    
    # MFCC (Mel-Frequency Cepstral Coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    features.extend(mfcc_mean)
      
    return np.array(features)


def preprocessing():
    global X, Y

    X_file = os.path.join(model_folder, "X.txt.npy")
    Y_file = os.path.join(model_folder, "Y.txt.npy")

    if os.path.exists(X_file) and os.path.exists(Y_file):
        X = np.load(X_file)
        Y = np.load(Y_file)
        print("X and Y arrays loaded successfully.")
    else:
        X = []  # input features
        Y = []  # output labels

        for root, dirs, files in os.walk(filename):
            name = os.path.basename(root)
            if name not in categories:
                continue
            
            for file in files:

                if file.endswith(('.mp3', '.wav')):

                    name = os.path.basename(root)   # folder name = class label
                    file_path = os.path.join(root, file)

                    print(f"Loading category: {name}")
                    print(file_path)

                    # Extract MFCC features
                    features = extract_features(file_path)

                    # Append features
                    X.append(features)

                    # Convert class name into numeric index
                    Y.append(categories.index(name))

        X = np.array(X)
        Y = np.array(Y)

        np.save(X_file, X)
        np.save(Y_file, Y)

    text.insert(END, f"Feature Matrix Shape: {X.shape}\n")
    text.insert(END, f"Label Vector Shape: {Y.shape}\n\n")

def train():
    global x_train, x_test, y_train, y_test
    from sklearn.model_selection import train_test_split

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    text.insert(END, f"X_train shape: {x_train.shape}\n\n")
    text.insert(END, f"y_train shape: {y_train.shape}\n\n")



all_overall_metrics = []
all_per_class_metrics = []

def calculateMetrics(algorithm, predict, testY, labels, plot_confusion=True):

    # =========================
    # 1. Overall Metrics
    # =========================
    a = accuracy_score(testY, predict) * 100
    p = precision_score(testY, predict, average='macro', zero_division=0) * 100
    r = recall_score(testY, predict, average='macro', zero_division=0) * 100
    f = f1_score(testY, predict, average='macro', zero_division=0) * 100

    overall_metrics = {
        "Algorithm": algorithm,
        "Accuracy": a,
        "Precision": p,
        "Recall": r,
        "F1 Score": f
    }

    all_overall_metrics.append(overall_metrics)

    text.insert(END,f"\n{algorithm} Accuracy: {a:.2f}\n")
    text.insert(END,f"{algorithm} Precision: {p:.2f}\n")
    text.insert(END,f"{algorithm} Recall: {r:.2f}\n")
    text.insert(END,f"{algorithm} F1 Score: {f:.2f}\n")


    # =========================
    # 2. Per-Class Metrics
    # =========================
    report = classification_report(testY, predict, target_names=labels, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(testY, predict)

    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    text.insert(END, f"\n{algorithm} Classification Report:\n")
    text.insert(END,"{:<10} {:<9} {:<9} {:<9} {:<9}".format(
        "Class", "Accuracy", "Precision", "Recall", "F1 Score\n"
    ))

    for i, class_name in enumerate(labels):

        acc = class_accuracy[i]
        prec = report[class_name]["precision"]
        rec = report[class_name]["recall"]
        f1 = report[class_name]["f1-score"]

        per_class_metrics = {
            "Algorithm": algorithm,
            "Class": class_name,
            "Accuracy": acc * 100,
            "Precision": prec * 100,
            "Recall": rec * 100,
            "F1 Score": f1 * 100
        }

        all_per_class_metrics.append(per_class_metrics)

        text.insert(END, "{:<10} {:<9.4f} {:<9.4f} {:<9.4f} {:<9.4f}\n".format(
            class_name, acc, prec, rec, f1
        ))


    # =========================
    # 3. Confusion Matrix Plot
    # =========================
    if plot_confusion:
        plt.figure(figsize=(6,5))
        sns.heatmap(conf_matrix, annot=True, fmt="d",
                    cmap="Blues",
                    xticklabels=labels,
                    yticklabels=labels)

        plt.title(f"{algorithm} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()


def model1():
    global model_path, sgd_model, y_pred ,sgd_path
    from sklearn.linear_model import SGDClassifier

    sgd_path = 'models/SGDClassifier.pkl'

    if os.path.exists(sgd_path):
        sgd_model = joblib.load(sgd_path)
        y_pred = sgd_model.predict(x_test)
        text.insert(END, "loaded")
        calculateMetrics('SGD Classifier', y_pred, y_test, categories)

    else:
        sgd_model = SGDClassifier(max_iter=1000, random_state=42)
        sgd_model.fit(x_train, y_train)

        # Save trained model
        joblib.dump(sgd_model, sgd_path)

        # Predict after training
        y_pred = sgd_model.predict(x_test)
        calculateMetrics('SGD Classifier', y_pred, y_test, categories)

def model2():
    global model_path, y_pred , lr_model, lr_path
    from sklearn.linear_model import LogisticRegression
    import os
    import joblib

    # Logistic Regression model path
    lr_path = 'models/LogisticRegression.pkl'

    if os.path.exists(lr_path):
        # Load existing model
        lr_model = joblib.load(lr_path)
        y_pred = lr_model.predict(x_test)
        
        text.insert(END, "Logistic Regression model loaded\n")
        calculateMetrics('LogisticRegression', y_pred, y_test, categories)

    else:
        # Train new model
        lr_model = LogisticRegression(C=0.00000001)
        lr_model.fit(x_train, y_train)

        # Save trained model
        joblib.dump(lr_model, lr_path)

        # Predict after training
        y_pred = lr_model.predict(x_test)
        calculateMetrics('LogisticRegression', y_pred, y_test, categories)


def model3():
    global model_path,svm_path, y_pred, svm_model, dt_path
    from sklearn.svm import SVC
    import os
    import joblib

    # SVM model path
    svm_path = 'models/SVMClassifier.pkl'

    if os.path.exists(svm_path):
        # Load existing model
        svm_model = joblib.load(svm_path)
        y_pred = svm_model.predict(x_test)

        text.insert(END, "SVM model loaded\n")
        calculateMetrics('SVM', y_pred, y_test, categories)

    else:
        # Train new model
        svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        svm_model.fit(x_train, y_train)

        # Save trained model
        joblib.dump(svm_model, svm_path)

        # Predict after training
        y_pred = svm_model.predict(x_test)
        calculateMetrics('SVM', y_pred, y_test, categories)

def model4():
    global model_path, y_pred, extra_model, extra_path
    from sklearn.ensemble import ExtraTreesClassifier

    extra_path = 'models/ExtraTreesClassifier.pkl'

    if os.path.exists(extra_path):
        extra_model = joblib.load(extra_path)
        y_pred = extra_model.predict(x_test)
        calculateMetrics('Extra Trees', y_pred, y_test, categories)

    else:
        extra_model = ExtraTreesClassifier(random_state=42)
        extra_model.fit(x_train, y_train)

        # Save trained model
        joblib.dump(extra_model, extra_path)

        # Predict after training
        y_pred = extra_model.predict(x_test)
        calculateMetrics('Extra Trees', y_pred, y_test, categories)

import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio
import numpy as np


def predict_audio():

    global model_path, pred_idx, pred_label, features, predicted_class, model

    model_path = os.path.join(model_folder, "ExtraTreesClassifier.pkl")

    model = joblib.load(model_path)

    path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[("Audio Files", "*.wav *.mp3")]
    )

    if not path:
        return

    # Load audio
    y, sr = librosa.load(path, sr=None)

    # Extract MFCC features
    features = extract_features(path).reshape(1, -1)

    # Safety check
    if features.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Feature vector length {features.shape[1]} does not match "
            f"model expected {model.n_features_in_} features."
        )

    # Predict
    pred_idx = model.predict(features)[0]
    pred_label = categories[pred_idx]

    # Display waveform with predicted label
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Predicted Class: {pred_label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    return pred_label





# ================= UI =================

main = Tk()
main.title("Road traffic conditions detection from environmental audio signals project")
main.geometry("1920x1080")

if os.path.exists(r"Road Traffic.jpg"):
    bg_img = ImageTk.PhotoImage(Image.open(r"Road Traffic.jpg").resize((1920, 1080)))
    bg_label = Label(main, image=bg_img)
    bg_label.place(x=0, y=0)
    bg_label.lower()
    main.bg_img = bg_img  

font = ('times', 15, 'bold')
font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

Label(main, text="Road traffic conditions detection from environmental audio signals project",
      bg="blue", fg="white",
      font=font, height=3, width=130).place(x=0, y=5)

Button(main, text="Dataset", command=uploadDataset, font=ff).place(x=20, y=150)
Button(main, text="Feature Extraction", command=preprocessing, font=ff).place(x=20, y=200)
Button(main, text="Train Test Split", command=train, font=ff).place(x=20, y=250)
Button(main, text="SGD Classifier", command=model1, font=ff).place(x=20, y=300)
Button(main, text="LogisticRegression", command=model2, font=ff).place(x=20, y=350)
Button(main, text="Support Vector Classifier", command=model3, font=ff).place(x=20, y=400)
Button(main, text="Extra Trees Classifier", command=model4, font=ff).place(x=20, y=450)
Button(main, text="Prediction", command=predict_audio, font=ff).place(x=20, y=500)

text = Text(main, height=25, width=100, font=font1)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=330, y=100)

main.mainloop()
