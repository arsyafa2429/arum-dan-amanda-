# Titanic Survival Prediction 🚢  
**Kelompok N**  
- Arum Ayu Lestari (NIM: 24293010)  
- Amanda Fitria Arsyafa (NIM: 24293007)

## 📌 Deskripsi Proyek

Proyek ini bertujuan untuk memprediksi keselamatan penumpang Titanic berdasarkan data dari Kaggle. Model prediktif dibuat menggunakan Python dan algoritma machine learning seperti Random Forest. Referensi utama proyek ini berasal dari:

- [Data Kompetisi Kaggle - Titanic](https://www.kaggle.com/competitions/titanic/data)  
- [Notebook oleh Alexis Cook](https://www.kaggle.com/code/alexisbcook/titanic-tutorial)

## 🧠 Metode

Model dibangun menggunakan langkah-langkah berikut:
1. Membaca dan membersihkan data (handling missing values, encoding).
2. Training model Random Forest.
3. Evaluasi akurasi pada validation set.
4. Prediksi data test dan membuat file `submission.csv`.

## 📦 Kode Python Terintegrasi

Seluruh proses di bawah ini dapat disimpan sebagai file `titanic_survival_prediction.py` dan dijalankan langsung.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 2. Preprocessing function
def preprocess(df):
    df = df.copy()
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    return df

# 3. Prepare training data
X = preprocess(train.drop('Survived', axis=1))
y = train['Survived']

# 4. Prepare test data
X_test = preprocess(test)

# 5. Split training & validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate model
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Akurasi model: {accuracy:.2f}")

# 8. Predict test set
test_preds = model.predict(X_test)

# 9. Buat file submission
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_preds
})
submission.to_csv('submission.csv', index=False)
print("File submission.csv berhasil dibuat.")
