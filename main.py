import pandas as pd
import glob
import os
import numpy as np

from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# DATA PATH
# =========================================================

data_path = "./harth-ml-experiments/harth"

# =========================================================
# SUBJECT SPLIT
# Last 9 subjects are test: S030-S038
# Others are train
# =========================================================

test_subjects = [
    "S030.csv", "S031.csv", "S032.csv",
    "S033.csv", "S034.csv", "S035.csv",
    "S036.csv", "S037.csv", "S038.csv"
]

# Tüm CSV dosyalarını bul
all_files = glob.glob(os.path.join(data_path, "*.csv"))
all_filenames = [os.path.basename(f) for f in all_files]

train_subjects = [
    f for f in all_filenames
    if f not in test_subjects
]

print("Train subjects:")
print(train_subjects)

print("\nTest subjects:")
print(test_subjects)

# =========================================================
# READ DATA
# =========================================================

dfs = []

for file_path in all_files:

    filename = os.path.basename(file_path)

    df = pd.read_csv(file_path)
    df["source_file"] = filename

    print(filename, "->", df.shape)

    dfs.append(df)

data = pd.concat(dfs, axis=0, ignore_index=True)

print("\nToplam veri boyutu:")
print(data.shape)

print("\nGenel label dağılımı:")
print(data["label"].value_counts())

# =========================================================
# TRAIN / TEST DATA
# =========================================================

train_data = data[data["source_file"].isin(train_subjects)]
test_data = data[data["source_file"].isin(test_subjects)]

print("\nTrain data shape:")
print(train_data.shape)

print("\nTest data shape:")
print(test_data.shape)

print("\nTrain label distribution:")
print(train_data["label"].value_counts())

print("\nTest label distribution:")
print(test_data["label"].value_counts())

# =========================================================
# FEATURE SETS
# =========================================================

feature_sets = {
    "Back Only": [
        "back_x", "back_y", "back_z"
    ],

    "Thigh Only": [
        "thigh_x", "thigh_y", "thigh_z"
    ],

    "Back + Thigh": [
        "back_x", "back_y", "back_z",
        "thigh_x", "thigh_y", "thigh_z"
    ]
}

# =========================================================
# WINDOW FEATURE FUNCTION
# =========================================================

def create_window_features(df, features, window_size=100, step_size=100):

    X_windows = []
    y_windows = []

    for start in range(0, len(df) - window_size + 1, step_size):

        window = df.iloc[start:start + window_size]

        feature_vector = []

        for col in features:

            values = window[col].values

            feature_vector.extend([
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
                np.median(values)
            ])

        label = mode(window["label"], keepdims=True).mode[0]

        X_windows.append(feature_vector)
        y_windows.append(label)

    return np.array(X_windows), np.array(y_windows)

# =========================================================
# TRAIN AND TEST FUNCTION
# =========================================================

def train_and_evaluate_model(model_name, features):

    print("\n===================================")
    print(model_name)
    print("Features:", features)
    print("===================================")

    # Train windows
    X_train_list = []
    y_train_list = []

    for subject in train_subjects:

        subject_df = train_data[
            train_data["source_file"] == subject
        ].reset_index(drop=True)

        X_sub, y_sub = create_window_features(
            subject_df,
            features,
            window_size=100,
            step_size=100
        )

        X_train_list.append(X_sub)
        y_train_list.append(y_sub)

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    # Test windows
    X_test_list = []
    y_test_list = []

    for subject in test_subjects:

        subject_df = test_data[
            test_data["source_file"] == subject
        ].reset_index(drop=True)

        X_sub, y_sub = create_window_features(
            subject_df,
            features,
            window_size=100,
            step_size=100
        )

        X_test_list.append(X_sub)
        y_test_list.append(y_sub)

    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)

    print("Window train shape:", X_train.shape)
    print("Window test shape :", X_test.shape)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
        output_dict=True
    )

    weighted_f1 = report["weighted avg"]["f1-score"]
    macro_f1 = report["macro avg"]["f1-score"]

    print("\nAccuracy:", accuracy)
    print("Weighted F1:", weighted_f1)
    print("Macro F1:", macro_f1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Weighted F1": weighted_f1,
        "Macro F1": macro_f1
    }

# =========================================================
# RUN ALL THREE MODELS
# =========================================================

results = []

for model_name, features in feature_sets.items():

    result = train_and_evaluate_model(
        model_name,
        features
    )

    results.append(result)

# =========================================================
# FINAL COMPARISON TABLE
# =========================================================

results_df = pd.DataFrame(results)

results_df["Accuracy (%)"] = results_df["Accuracy"] * 100
results_df["Weighted F1 (%)"] = results_df["Weighted F1"] * 100
results_df["Macro F1 (%)"] = results_df["Macro F1"] * 100

print("\n===================================")
print("FINAL SENSOR COMPARISON")
print("===================================")

print(results_df[
    [
        "Model",
        "Accuracy (%)",
        "Weighted F1 (%)",
        "Macro F1 (%)"
    ]
])

# CSV olarak kaydet
output_path = os.path.join(data_path, "sensor_comparison_results.csv")
results_df.to_csv(output_path, index=False)

print("\nSonuç dosyası kaydedildi:")
print(output_path)
