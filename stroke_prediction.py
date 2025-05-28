import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages 
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, ttest_ind

pdf_pages = PdfPages('plots.pdf')

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df = df[df['gender'] != 'Other']
df['bmi'].fillna(df['bmi'].median(), inplace=True)
df.drop(columns=['id'], inplace=True)
df['stroke'] = df['stroke'].astype(int)

numeric_cols = ['age', 'avg_glucose_level', 'bmi']
categorical_cols = [c for c in df.columns if c not in numeric_cols + ['stroke']]


print("*** DESCRIPTIVE STATISTICS ***")
for col in numeric_cols:
    print(f"\n*** {col.upper()} SUMMARY ***")
    print(df[col].describe())
    counts, bins = np.histogram(df[col].dropna(), bins=20)
    print(f"Histogram bins:\n {bins}\nCounts:\n {counts}")

    fig = plt.figure(figsize=(8,5))
    sns.histplot(df[col].dropna(), bins=bins, kde=True)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    pdf_pages.savefig(fig)
    plt.close(fig)

for col in categorical_cols:
    counts = df[col].value_counts(dropna=False)
    print(f"\n*** {col.upper()} COUNTS ***")
    print(counts.to_frame('count'))

    fig = plt.figure(figsize=(8,5))
    sns.countplot(y=col, data=df, order=counts.index, palette='viridis')
    plt.title(f"Bar chart of {col}")
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.tight_layout()
    pdf_pages.savefig(fig)
    plt.close(fig)


stroke0 = df[df['stroke'] == 0]
stroke1 = df[df['stroke'] == 1]

print("\n*** NUMERIC FEATURES VS. STROKE STATUS ***")
for col in numeric_cols:
    print(f"\n*** {col.upper()} BY STROKE STATUS ***")
    print("No Stroke:", stroke0[col].describe())
    print("Stroke:   ", stroke1[col].describe())

    stat, p = ttest_ind(stroke0[col].dropna(), stroke1[col].dropna())
    print(f"T-test for {col}: t-statistic = {stat:.3f}, p-value = {p:.3f}")
    if p < 0.05:
        print(f"  -> There is a significant difference in {col} between stroke and non-stroke groups.")
    else:
        print(f"  -> No significant difference in {col} between stroke and non-stroke groups.")

    fig = plt.figure(figsize=(8,5))
    if col == 'age':
        min_val = min(stroke0[col].min(), stroke1[col].min())
        max_val = max(stroke0[col].max(), stroke1[col].max())
        shared_bins = np.linspace(min_val, max_val, 21)
        c0, _ = np.histogram(stroke0[col], bins=shared_bins)
        c1, _ = np.histogram(stroke1[col], bins=shared_bins)
        print("Bins:", shared_bins)
        print("No Stroke counts:", c0)
        print("Stroke counts:   ", c1)
        sns.histplot(stroke0[col], bins=shared_bins, color='skyblue', label='No Stroke', alpha=0.6, kde=True)
        sns.histplot(stroke1[col], bins=shared_bins, color='salmon', label='Stroke', alpha=0.6, kde=True)
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.legend()
    else:
        data0 = stroke0[col].dropna()
        data1 = stroke1[col].dropna()
        if not data0.empty:
             print("No Stroke quartiles:", np.percentile(data0, [0,25,50,75,100]))
        else:
             print("No Stroke quartiles: No data")
        if not data1.empty:
            print("Stroke quartiles:   ", np.percentile(data1, [0,25,50,75,100]))
        else:
            print("Stroke quartiles: No data")
        sns.violinplot(x='stroke', y=col, data=df, palette='muted')
        plt.xticks([0,1], ['No Stroke','Stroke'])
        plt.ylabel(col)
    plt.title(f"{col.replace('_',' ').capitalize()} by Stroke Status")
    plt.tight_layout()
    pdf_pages.savefig(fig)
    plt.close(fig)


print("\n*** CATEGORICAL FEATURES VS. STROKE STATUS ***")
for col in categorical_cols:
    ct = pd.crosstab(df['stroke'], df[col])
    print(f"\n*** Crosstab of {col} by Stroke ***")
    print(ct)

    chi2, p_val, dof, expected = chi2_contingency(ct)
    print(f"Chi-square test for {col}: Chi2 = {chi2:.3f}, p-value = {p_val:.3f}")
    if p_val < 0.05:
        print(f"  -> There is a significant association between {col} and stroke status.")
    else:
        print(f"  -> No significant association between {col} and stroke status.")

    ct_norm = ct.apply(lambda r: r/r.sum(), axis=1)
    print(f"\n*** Proportions of {col} by Stroke ***")
    print(ct_norm)

    fig = plt.figure(figsize=(8,5))
    ct_norm.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='Pastel1')
    plt.title(f"Stroke vs {col}")
    plt.xlabel("Stroke (0=No, 1=Yes)")
    plt.ylabel("Proportion")
    plt.legend(title=col, bbox_to_anchor=(1.05,1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    pdf_pages.savefig(fig)
    plt.close(fig)

corr = df[numeric_cols + ['stroke']].corr()
print("\n*** CORRELATION MATRIX ***")
print(corr)
fig = plt.figure(figsize=(8,7))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap", pad=20)
plt.tight_layout()
pdf_pages.savefig(fig)
plt.close(fig)


df_log = pd.get_dummies(df, drop_first=True)
X_log = df_log.drop(columns=['stroke'])
y_log = df_log['stroke']

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_log, y_log, test_size=0.3, random_state=42, stratify=y_log
)

scaler = StandardScaler()
X_train_l_scaled = scaler.fit_transform(X_train_l)
X_test_l_scaled = scaler.transform(X_test_l)

log_model = LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced')
log_model.fit(X_train_l_scaled, y_train_l)
y_pred_l = log_model.predict(X_test_l_scaled)
y_proba_l = log_model.predict_proba(X_test_l_scaled)[:,1]

print("\n*** LOGISTIC REGRESSION REPORT ***")
print(classification_report(y_test_l, y_pred_l, zero_division=0))
acc_l = accuracy_score(y_test_l, y_pred_l)
auc_l = roc_auc_score(y_test_l, y_proba_l)
print(f"Logistic Accuracy: {acc_l:.3f}")
print(f"Logistic ROC AUC:   {auc_l:.3f}")

odds_ratios = pd.DataFrame({
    'feature': X_log.columns,
    'coefficient': log_model.coef_[0],
    'odds_ratio': np.exp(log_model.coef_[0])
}).sort_values(by='odds_ratio', ascending=False)
print("\nLogistic Regression Odds Ratios:")
print(odds_ratios)

# ROC Curve - Logistic Regression
fig = plt.figure(figsize=(8,6))
fpr_l, tpr_l, _ = roc_curve(y_test_l, y_proba_l)
plt.plot(fpr_l, tpr_l, label=f"AUC = {auc_l:.2f}")
plt.plot([0,1],[0,1],'--',color='gray',alpha=0.7)
plt.title("ROC Curve – Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
pdf_pages.savefig(fig)
plt.close(fig)

# Precision-Recall Curve - Logistic Regression
fig = plt.figure(figsize=(8,6))
precision_l, recall_l, _ = precision_recall_curve(y_test_l, y_proba_l)
pr_auc_l = auc(recall_l, precision_l)
plt.plot(recall_l, precision_l, label=f"PR AUC = {pr_auc_l:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve – Logistic Regression")
plt.legend(loc='lower left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
pdf_pages.savefig(fig)
plt.close(fig)

# Confusion Matrix - Logistic Regression
fig = plt.figure(figsize=(6,5))
cm_l = confusion_matrix(y_test_l, y_pred_l)
sns.heatmap(cm_l, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted No Stroke', 'Predicted Stroke'],
            yticklabels=['Actual No Stroke', 'Actual Stroke'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Logistic Regression')
plt.tight_layout()
pdf_pages.savefig(fig)
plt.close(fig)

X_rf, y_rf = X_log, y_log

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf, y_rf, test_size=0.3, stratify=y_rf, random_state=42
)

rf = RandomForestClassifier(random_state=42, class_weight='balanced')
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(rf, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
grid.fit(X_train_rf, y_train_rf)

best_rf = grid.best_estimator_
print("\nBest RF params:", grid.best_params_)
print(f"Best RF CV ROC AUC: {grid.best_score_:.3f}")

y_pred_rf = best_rf.predict(X_test_rf)
y_proba_rf = best_rf.predict_proba(X_test_rf)[:,1]

print("\n*** RANDOM FOREST REPORT ***")
print(classification_report(y_test_rf, y_pred_rf, zero_division=0))
acc_rf = accuracy_score(y_test_rf, y_pred_rf)
auc_rf = roc_auc_score(y_test_rf, y_proba_rf)
print(f"RF Accuracy:   {acc_rf:.3f}")
print(f"RF ROC AUC:    {auc_rf:.3f}")

feature_importances = pd.DataFrame({
    'feature': X_rf.columns,
    'importance': best_rf.feature_importances_
}).sort_values(by='importance', ascending=False)
print("\nRandom Forest Feature Importances:")
print(feature_importances)

# Feature Importances - Random Forest
fig = plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feature_importances, palette='mako')
plt.title('Random Forest Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
pdf_pages.savefig(fig)
plt.close(fig)

# ROC Curve - Random Forest
fig = plt.figure(figsize=(8,6))
fpr_rf, tpr_rf, _ = roc_curve(y_test_rf, y_proba_rf)
plt.plot(fpr_rf, tpr_rf, label=f"AUC = {auc_rf:.2f}")
plt.plot([0,1],[0,1],'--',color='gray',alpha=0.7)
plt.title("ROC Curve – Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
pdf_pages.savefig(fig)
plt.close(fig)

# Precision-Recall Curve - Random Forest
fig = plt.figure(figsize=(8,6))
precision_rf, recall_rf, _ = precision_recall_curve(y_test_rf, y_proba_rf)
pr_auc_rf = auc(recall_rf, precision_rf)
plt.plot(recall_rf, precision_rf, label=f"PR AUC = {pr_auc_rf:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve – Random Forest")
plt.legend(loc='lower left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
pdf_pages.savefig(fig)
plt.close(fig)

# Confusion Matrix - Random Forest
fig = plt.figure(figsize=(6,5))
cm_rf = confusion_matrix(y_test_rf, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted No Stroke', 'Predicted Stroke'],
            yticklabels=['Actual No Stroke', 'Actual Stroke'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Random Forest')
plt.tight_layout()
pdf_pages.savefig(fig)
plt.close(fig)

print("\n*** MODEL PERFORMANCE COMPARISON ***")
performance_data = {
    'Metric': ['Accuracy', 'ROC AUC', 'PR AUC'],
    'Logistic Regression': [acc_l, auc_l, pr_auc_l],
    'Random Forest': [acc_rf, auc_rf, pr_auc_rf]
}
performance_df = pd.DataFrame(performance_data)
print(performance_df.set_index('Metric').round(3))

# ROC Curve Comparison
fig = plt.figure(figsize=(8,6))
plt.plot(fpr_l, tpr_l, label=f"Logistic Regression (AUC = {auc_l:.2f})", color='blue')
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.2f})", color='red')
plt.plot([0,1],[0,1],'--',color='gray',alpha=0.7)
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
pdf_pages.savefig(fig)
plt.close(fig)

pdf_pages.close()
print("\n--- Analysis Complete ---")