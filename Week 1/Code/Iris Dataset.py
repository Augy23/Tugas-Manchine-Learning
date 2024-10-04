# Impor library yang diperlukan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Memuat dataset
file_path = '/mnt/data/Iris.csv'
df = pd.read_csv(file_path)

# Menghapus kolom 'Id' dan memisahkan fitur dan target
df = df.drop(columns=['Id'])
X = df.drop(columns=['Species'])
y = df['Species']

# Encode label pada kolom target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Melatih model Logistic Regression
logreg_model = LogisticRegression(max_iter=200)
logreg_model.fit(X_train, y_train)
y_pred_logreg = logreg_model.predict(X_test)
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)

# Melatih model K-Nearest Neighbors (K-NN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)

# Menampilkan akurasi kedua model
print(f"Akurasi Logistic Regression: {logreg_accuracy}")
print(f"Akurasi K-NN: {knn_accuracy}")

