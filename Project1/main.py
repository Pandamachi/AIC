import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score, make_scorer
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout

# Read data
data = pd.read_csv('data2000.csv')

# map position to FWD, MID, DEF, GK
position_mapping = {
    "ST": "FWD", "CF": "FWD", "LW": "FWD", "RW": "FWD",
    "CM": "MID", "CDM": "MID", "CAM": "MID", "LM": "MID", "RM": "MID",
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
    "GK": "GK"
}
data["Position"] = data["Position"].map(position_mapping)

# label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["Position"])
X = data.drop(columns=["Position"])

# standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Randomly pick 1000 samples
X_test, y_test = resample(X_test, y_test, n_samples=1000, random_state=42)


# ========== use SMOTE to deal with data imbalance  problem ==========
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\n===== SMOTE done =====")
print(pd.Series(y_train_resampled).value_counts())
input("Press Enter to continue...\n")

# ========== Supervised Learning Models (using cross-validation) ==========
# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True),
    "LDA": LinearDiscriminantAnalysis()
}

# train models
results = []
for name, model in models.items():
    acc_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring="accuracy")
    precision_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring="precision_weighted")
    recall_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring="recall_weighted")
    f1_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring="f1_weighted")
    
    result = {
        "Model": name,
        "Accuracy": np.mean(acc_scores),
        "Precision": np.mean(precision_scores),
        "Recall": np.mean(recall_scores),
        "F1 Score": np.mean(f1_scores)
    }

    print(f"\n{name} (Cross-Validation Results):")
    print(f"Accuracy: {result['Accuracy']:.4f}")
    print(f"Precision: {result['Precision']:.4f}")
    print(f"Recall: {result['Recall']:.4f}")
    print(f"F1 Score: {result['F1 Score']:.4f}")
    input("Press Enter to continue...\n")

    results.append(result)

# ========== CNN ==========
def train_cnn(X_train, y_train, X_test, y_test):
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    cnn_model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(np.unique(y)), activation='softmax')
    ])

    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train_cnn, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    
    y_pred_cnn = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
    return y_pred_cnn

y_pred_cnn = train_cnn(X_train_resampled, y_train_resampled, X_test, y_test)

cnn_result = {
    "Model": "CNN",
    "Accuracy": accuracy_score(y_test, y_pred_cnn),
    "Precision": precision_score(y_test, y_pred_cnn, average='weighted', zero_division=0),
    "Recall": recall_score(y_test, y_pred_cnn, average='weighted', zero_division=0),
    "F1 Score": f1_score(y_test, y_pred_cnn, average='weighted', zero_division=0)
}
print(f"\nCNN Results:")
print(f"Accuracy: {cnn_result['Accuracy']:.4f}")
print(f"Precision: {cnn_result['Precision']:.4f}")
print(f"Recall: {cnn_result['Recall']:.4f}")
print(f"F1 Score: {cnn_result['F1 Score']:.4f}")
input("Press Enter to continue...\n")

results.append(cnn_result)

# ========== K-Means (PCA) ==========
pca = PCA(n_components=2)  # down to 2 dimensions for visualization
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42, n_init=10)
kmeans.fit(X_train_pca)
y_pred_kmeans = kmeans.predict(X_test_pca)

# calculate inertia and silhouette score
inertia = kmeans.inertia_
silhouette = silhouette_score(X_test_pca, y_pred_kmeans)

print("\nK-Means Clustering Results:")
print(f"Inertia: {inertia:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")
input("Press Enter to end...\n")

