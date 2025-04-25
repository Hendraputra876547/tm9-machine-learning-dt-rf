import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('Blabla.xlsx', sheet_name='Sheet1')
data = df[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'UMUR_TAHUN']]

print(' Data Awal '.center(75, '='))
print(data)
print('=' * 75, '\n')

# Grouping yang dibagi menjadi dua 
print(' Grouping Variable '.center(75, '='))
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values
print(' Data Variabel '.center(75, '='))
print(X)
print(' Data Kelas '.center(75, '='))
print(y)
print('=' * 75, '\n')

# Pembagian training dan testing data
print(' Split Data 20-80 '.center(75, '='))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(' Instance Variabel Data Training '.center(75, '='))
print(X_train)
print(' Instance Kelas Data Training '.center(75, '='))
print(y_train)
print(' Instance Variabel Data Testing '.center(75, '='))
print(X_test)
print(' Instance Kelas Data Testing '.center(75, '='))
print(y_test)
print('=' * 75, '\n')

# Permodelan Decision Tree
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)

# Prediksi Decision Tree
print(' Instance Prediksi Decision Tree '.center(75, '='))
y_pred = decision_tree.predict(X_test)
print(y_pred)
print('=' * 75, '\n')

# Prediksi Akurasi Decision Tree
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
print('Akurasi: ', accuracy, '%\n')

# Display Classification Report
print(' Classification Report Decision Tree '.center(75, '='))
print(classification_report(y_test, y_pred))

# Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: \n', cm)

# Visualisasi Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(decision_tree, filled=True, feature_names=data.columns[:-1], class_names=['0', '1'], rounded=True, node_ids=False, proportion=False, impurity=False)
plt.title('Decision Tree Visualization')
plt.show()

# =================== RANDOM FOREST =====================
print(' Random Forest '.center(75, '='))

rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(X_train, y_train)

# Prediksi Random Forest
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]  

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Akurasi
accuracy_rf = round(accuracy_score(y_test, y_pred_rf)* 100, 2)
precision_rf = round(classification_report(y_test, y_pred_rf, output_dict=True)['1']['precision'] * 100, 2)
recall_rf = round(classification_report(y_test, y_pred_rf, output_dict=True)['1']['recall'] * 100, 2)
f1_rf = round(classification_report(y_test, y_pred_rf, output_dict=True)['1']['f1-score'] * 100, 2)
auc_rf = round(roc_auc_score(y_test, y_proba_rf) * 100, 2)

# Print metrics
print('Confusion Matrix Random Forest:')
print(cm_rf, '\n')
print(f"Akurasi: ", accuracy_rf, "%")
print(f"Presisi: ", precision_rf, "%")
print(f"Recall: ", recall_rf, "%")
print(f"F1-Score: ", f1_rf, "%")
print(f"AUC-ROC: ", auc_rf, "%")
print('=' * 75, '\n')

# Visualisasi Random Forest (Feature Importance)
importances = rf_model.feature_importances_
features = data.columns[0:13]

plt.figure(figsize=(12, 6))
sns.barplot(x=importances, y=features, palette='viridis')
plt.title('Feature Importance Random Forest')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# ambil salah satu pohon, misal pohon ke-0
estimator = rf_model.estimators_[11]

plt.figure(figsize=(20, 10))
plot_tree(estimator, filled=True, feature_names=data.columns[:-1], class_names=['0', '1'], rounded=True, impurity=False)
plt.title('Visualisasi Fitur K Decision Tree dari Random Forest')
plt.show()

# === Menyimpan metrik hasil Decision Tree ===
y_proba_dt = decision_tree.predict_proba(X_test)[:, 1]

metrics_dt = {
    'Accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
    'Precision': round(classification_report(y_test, y_pred, output_dict=True)['1']['precision'] * 100, 2),
    'Recall': round(classification_report(y_test, y_pred, output_dict=True)['1']['recall'] * 100, 2),
    'F1-Score': round(classification_report(y_test, y_pred, output_dict=True)['1']['f1-score'] * 100, 2),
    'AUC-ROC': round(roc_auc_score(y_test, y_proba_dt) * 100, 2)
}

# === Menyimpan metrik hasil Random Forest ===
metrics_rf = {
    'Accuracy': accuracy_rf,
    'Precision': precision_rf,
    'Recall': recall_rf,
    'F1-Score': f1_rf,
    'AUC-ROC': auc_rf
}

# === Tampilkan metrik perbandingan ===
print('Perbandingan Kinerja Model'.center(75, '='))
print(pd.DataFrame([metrics_dt, metrics_rf], index=['Decision Tree', 'Random Forest']))
print('=' * 75, '\n')

# === Tampilkan Confusion Matrix side-by-side ===
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Decision Tree
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix - Decision Tree')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# Random Forest
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Confusion Matrix - Random Forest')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
plt.show()

# Hitung ROC Curve untuk Decision Tree
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
auc_dt = roc_auc_score(y_test, y_proba_dt)

# Hitung ROC Curve untuk Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)

# Plot ROC Curve
plt.figure(figsize=(10, 7))

# Decision Tree ROC
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})', color='blue')

# Random Forest ROC
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='green')

# Garis baseline (random guess)
plt.plot([0, 1], [0, 1], 'k--')

# Setting tampilan
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# === Visualisasi Perbandingan Metrik ke Histogram ===
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

dt_scores = [metrics_dt[m] for m in metrics]
rf_scores = [metrics_rf[m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(12, 7))
plt.bar(x - width/2, dt_scores, width, label='Decision Tree', color='skyblue')
plt.bar(x + width/2, rf_scores, width, label='Random Forest', color='lightgreen')

plt.ylabel('Score (%)')
plt.title('Perbandingan Kinerja Model Decision Tree vs Random Forest')
plt.xticks(x, metrics)
plt.ylim(0, 110)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Coba Input
def check_input(value):
    while True:
        val = input(value)
        if val.upper() == 'Y' or val.upper() == 'YES' or val.upper() == 'y':
            return 1
        elif val.upper() == 'N' or val.upper() == 'NO' or val.upper() == 'n':
            return 0
        else:
            print('Input tidak valid. Silakan masukkan Y atau N.')
            continue

def check_gender(value):
    while True:
        val = input(value)
        if val.upper() == 'P' or val.upper() == 'Perempuan' or val.upper() == 'p' or val.upper() == 'PEREMPUAN' or val.upper() == 'perempuan':
            return 1
        elif val.upper() == 'L' or val.upper() == 'Laki-laki' or val.upper() == 'l' or val.upper() == 'LAKI-LAKI' or val.upper() == 'laki-laki':
            return 0
        else:
            print('Input tidak valid. Silakan masukkan P atau L.')
            continue

print(' Contoh Input '.center(75, '='))
A = int(input('Umur Pasien: '))
print('Isi Jenis Kelamin dengan P jika perempuan dan L jika laki-laki')
B = check_gender('Jenis Kelamin Pasien: ')
print('Isi dengan Y jika mengalami dan N jika tidak')
C = check_input('Apakah Pasien Mengalami C? : ')
D = check_input('Apakah Pasien Mengalami D? : ')
E = check_input('Apakah Pasien Mengalami E? : ')
F = check_input('Apakah Pasien Mengalami F? : ')
G = check_input('Apakah Pasien Mengalami G? : ')
H = check_input('Apakah Pasien Mengalami H? : ')
I = check_input('Apakah Pasien Mengalami I? : ')
J = check_input('Apakah Pasien Mengalami J? : ')
K = check_input('Apakah Pasien Mengalami K? : ')
L = check_input('Apakah Pasien Mengalami L? : ')
M = check_input('Apakah M? : ')

umur_k = 0
A_k = 0

if A<21:
    A_k = 1
if A>20 and A<31:
    A_k = 2
if A>30 and A<41:
    A_k = 3
if A>40 and A<51:
    A_k = 4
if A>50:
    A_k = 5
print('Kode umur pasien adalah ', A_k)

Train = [A_k, B, C, D, E, F, G, H, I, J, K, L, M]
print(Train)

test = pd.DataFrame(Train).T
predtest = decision_tree.predict(test)

if predtest == 1:
    print('Prediksi Decision Tree: Pasien Positif')
elif predtest == 0:
    print('Prediksi Decision Tree: Pasien Negatif')
else:
    print('Data tidak valid')

# Prediksi dengan Random Forest
predtest_rf = rf_model.predict(test)

if predtest_rf == 1:
    print('Prediksi Random Forest: Pasien Positif\n')
elif predtest_rf == 0:
    print('Prediksi Random Forest: Pasien Negatif\n')
else:
    print('Data tidak valid\n')