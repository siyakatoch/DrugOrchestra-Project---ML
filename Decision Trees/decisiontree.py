import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
print('started')
class CombinedNPYDataset:
    def __init__(self, file_path):
        data = np.load(file_path)
        self.features = data[:, :-1]
        self.labels = data[:, -1]

def process_classification_dataset(file_path, n_splits=3):
    dataset = CombinedNPYDataset(file_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataset.features)
    pca = PCA(n_components=10)  
    X_pca = pca.fit_transform(X_scaled)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores, auprc_scores = [], []

    for train_index, test_index in skf.split(X_pca, dataset.labels):
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = dataset.labels[train_index], dataset.labels[test_index]

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:, 1]

        auc_scores.append(roc_auc_score(y_test, y_pred))
        auprc_scores.append(average_precision_score(y_test, y_pred))

    auc_mean, auc_std = np.mean(auc_scores), np.std(auc_scores)
    auprc_mean, auprc_std = np.mean(auprc_scores), np.std(auprc_scores)
    return auc_mean, auc_std, auprc_mean, auprc_std

def process_regression_dataset(file_path):
    dataset = CombinedNPYDataset(file_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataset.features)
    pca = PCA(n_components=10) 
    X_pca = pca.fit_transform(X_scaled)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    mse_scores, scc_scores = [], []

    for train_index, test_index in kf.split(X_pca):
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = dataset.labels[train_index], dataset.labels[test_index]

        reg = DecisionTreeRegressor()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        scc_scores.append(spearmanr(y_test, y_pred)[0])

    mse_mean, mse_std = np.mean(mse_scores), np.std(mse_scores)
    scc_mean, scc_std = np.mean(scc_scores), np.std(scc_scores)
    return mse_mean, mse_std, scc_mean, scc_std


base_dir = r"C:\Users\Akin\Desktop\datas\data"
classification_datasets = ["drugbank", "repurposing_hub", "sider", "stitch", "offside"]

regression_datasets = ["ccle", "gdsc", "pdx"]  

for dataset_name in classification_datasets:
    file_path = f"{base_dir}\\{dataset_name}\\data.npy"
    auc_mean, auc_std, auprc_mean, auprc_std = process_classification_dataset(file_path)  # Adjusted to unpack all returned values
    print(f"Classification: {dataset_name}, AUC: {auc_mean:.4f} (±{auc_std:.4f}), AUPRC: {auprc_mean:.4f} (±{auprc_std:.4f})")


print('classification ended')
for dataset_name in regression_datasets:
    file_path = f"{base_dir}\\{dataset_name}\\data.npy"
    mse_mean, mse_std, scc_mean, scc_std = process_regression_dataset(file_path)  # Adjusted to unpack all returned values
    print(f"Regression: {dataset_name}, MSE: {mse_mean:.4f} (±{mse_std:.4f}), SCC: {scc_mean:.4f} (±{scc_std:.4f})")



'''
Classification Dataset: drugbank, AUC: 0.7430, AUPRC: 0.3091
Classification Dataset: repurposing_hub, AUC: 0.7165, AUPRC: 0.2648
Classification Dataset: sider, AUC: 0.6756, AUPRC: 0.2527
Classification Dataset: stitch, AUC: 0.9180, AUPRC: 0.6632
Classification Dataset: offside, AUC: 0.7997, AUPRC: 0.5831
Regression Dataset: ccle, MSE: 0.5795, SCC: 0.6578
Regression Dataset: gdsc, MSE: 0.7622, SCC: 0.5939
Regression Dataset: pdx, MSE: 1.5199, SCC: 0.3265
'''