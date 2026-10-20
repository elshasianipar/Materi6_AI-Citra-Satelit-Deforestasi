from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT/"data"/"fitur_lahan.csv")
features = ["ndvi_sebelum","ndvi_sesudah","nbr_sebelum","nbr_sesudah","jarak_jalan_km"]
X, y = df[features], df["kelas"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=7, stratify=y
)
model = RandomForestClassifier(n_estimators=100, random_state=7)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Jumlah sampel simulasi: {len(df)}")
print(f"Accuracy: {accuracy_score(y_test, pred):.2f}")
print("\nClassification report:")
print(classification_report(y_test, pred, zero_division=0))
sample = pd.DataFrame([{
    "ndvi_sebelum": .80, "ndvi_sesudah": .16,
    "nbr_sebelum": .62, "nbr_sesudah": .03, "jarak_jalan_km": 3.4
}])
print("Prediksi contoh simulasi:", model.predict(sample)[0])
print("Hasil ini hanya demonstrasi, bukan bukti penyebab perubahan lahan.")
