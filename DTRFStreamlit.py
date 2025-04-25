import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = pd.read_excel('Blabla.xlsx', sheet_name='Sheet1')
data = df[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'UMUR_TAHUN']]

X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Model
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)

rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(X_train, y_train)

# Prediksi
y_pred_dt = decision_tree.predict(X_test)
y_proba_dt = decision_tree.predict_proba(X_test)[:, 1]

y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_rf = confusion_matrix(y_test, y_pred_rf)

# ROC Curve
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

# METRICS SCORES
dt_acc = accuracy_score(y_test, y_pred_dt) * 100
dt_prec = precision_score(y_test, y_pred_dt) * 100
dt_rec = recall_score(y_test, y_pred_dt) * 100
dt_f1 = f1_score(y_test, y_pred_dt) * 100

rf_acc = accuracy_score(y_test, y_pred_rf) * 100
rf_prec = precision_score(y_test, y_pred_rf) * 100
rf_rec = recall_score(y_test, y_pred_rf) * 100
rf_f1 = f1_score(y_test, y_pred_rf) * 100

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
dt_scores = [dt_acc, dt_prec, dt_rec, dt_f1]
rf_scores = [rf_acc, rf_prec, rf_rec, rf_f1]
x = np.arange(len(metrics))
width = 0.35

# STREAMLIT START
st.title("Aplikasi Prediksi Pasien | Decision Tree & Random Forest")

# Plot Decision Tree
st.subheader("Visualisasi Decision Tree")
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(decision_tree, 
          feature_names=data.columns[:13], 
          class_names=['Negatif', 'Positif'], 
          filled=True, 
          rounded=True, 
          fontsize=1, 
          ax=ax)
st.pyplot(fig)

# Plot Random Forest (Tree pertama)
st.subheader("Visualisasi Tree Pertama di Random Forest")
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(rf_model.estimators_[0], 
          feature_names=data.columns[:13], 
          class_names=['Negatif', 'Positif'], 
          filled=True, 
          rounded=True, 
          fontsize=1, 
          ax=ax)
st.pyplot(fig)

# Visualisasi ROC Curve
st.subheader("ROC Curve")
fig, ax = plt.subplots()
ax.plot(fpr_dt, tpr_dt, label="Decision Tree", color='blue')
ax.plot(fpr_rf, tpr_rf, label="Random Forest", color='green')
ax.plot([0,1], [0,1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
st.pyplot(fig)

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Decision Tree')
axes[0].set_xlabel('Prediksi')
axes[0].set_ylabel('Aktual')

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Random Forest')
axes[1].set_xlabel('Prediksi')
axes[1].set_ylabel('Aktual')
st.pyplot(fig)

# Feature Importance
st.subheader("Feature Importance Random Forest")
importances = rf_model.feature_importances_
features = data.columns[0:13]
fig, ax = plt.subplots()
sns.barplot(x=importances, y=features, palette='viridis', ax=ax)
ax.set_title('Feature Importance Random Forest')
st.pyplot(fig)

# Visualisasi Perbandingan Kinerja Model
st.subheader("Perbandingan Kinerja Model Decision Tree vs Random Forest")
fig, ax = plt.subplots(figsize=(12, 7))
ax.bar(x - width/2, dt_scores, width, label='Decision Tree', color='skyblue')
ax.bar(x + width/2, rf_scores, width, label='Random Forest', color='lightgreen')
ax.set_ylabel('Score (%)')
ax.set_title('Perbandingan Kinerja Model Decision Tree vs Random Forest')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 110)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot(fig)

# Sidebar buat Prediksi
st.sidebar.header("Input Data Pasien")

A = st.sidebar.number_input('Umur Pasien', min_value=0, max_value=100, value=25)
B = st.sidebar.radio('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
C = st.sidebar.radio('Apakah Mengalami C?', ['Y', 'N'])
D = st.sidebar.radio('Apakah Mengalami D?', ['Y', 'N'])
E = st.sidebar.radio('Apakah Mengalami E?', ['Y', 'N'])
F = st.sidebar.radio('Apakah Mengalami F?', ['Y', 'N'])
G = st.sidebar.radio('Apakah Mengalami G?', ['Y', 'N'])
H = st.sidebar.radio('Apakah Mengalami H?', ['Y', 'N'])
I = st.sidebar.radio('Apakah Mengalami I?', ['Y', 'N'])
J = st.sidebar.radio('Apakah Mengalami J?', ['Y', 'N'])
K = st.sidebar.radio('Apakah Mengalami K?', ['Y', 'N'])
L = st.sidebar.radio('Apakah Mengalami L?', ['Y', 'N'])
M = st.sidebar.radio('Apakah Mengalami M?', ['Y', 'N'])

A_k = 0
if A < 21:
    A_k = 1
elif A < 31:
    A_k = 2
elif A < 41:
    A_k = 3
elif A < 51:
    A_k = 4
else:
    A_k = 5

gender = 1 if B == 'Perempuan' else 0

def yn_to_bin(val):
    return 1 if val == 'Y' else 0

input_data = [A_k, gender, yn_to_bin(C), yn_to_bin(D), yn_to_bin(E), yn_to_bin(F),
              yn_to_bin(G), yn_to_bin(H), yn_to_bin(I), yn_to_bin(J),
              yn_to_bin(K), yn_to_bin(L), yn_to_bin(M)]

# Tombol Prediksi
if st.sidebar.button("Prediksi"):
    test_df = pd.DataFrame([input_data])

    pred_dt = decision_tree.predict(test_df)[0]
    pred_rf = rf_model.predict(test_df)[0]

    st.sidebar.subheader("Hasil Prediksi")
    st.sidebar.write(f"**Decision Tree:** {'Positif' if pred_dt == 1 else 'Negatif'}")
    st.sidebar.write(f"**Random Forest:** {'Positif' if pred_rf == 1 else 'Negatif'}")
