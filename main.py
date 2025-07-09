import numpy as np
from models import PDProx1, SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataset.loader import LoadData
import pandas as pd

# ------------------------- Load Data ------------------------
scaler = StandardScaler()
dataset = 'adult'  # or 'breast_cancer', 'ionosphere', etc.
loader = LoadData(scaler, dataset)
X, y = loader.data, loader.target
y = 2 * y - 1  # convert to {-1, +1}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------- Param Grids ------------------------
c_grid = [0.001, 0.01, 0.1, 1]
lambda_grid = [0.001, 0.01, 0.1, 1]
eta_values = [0.001,0.005,0.1, 0.5, 1, 2]

# ------------------------- Metric Helper ------------------------
def get_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }

# ------------------------- Grid Search ------------------------
def grid_search_pdprox(X_train, y_train, X_test, y_test, eta, iter=300):
    best_score = -1
    best_model = None
    for C in c_grid:
        for lmbda in lambda_grid:
            model = PDProx1.PDProx(X_train, y_train, C=C, lambda_=lmbda, eta=eta, iter=iter)
            model.train()
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            if acc > best_score:
                best_score = acc
                best_model = model
    return best_model

def grid_search_svc(X_train, y_train, X_test, y_test, iter=-1):
    best_score = -1
    best_model = None
    for C in c_grid:
        print(C)
        model = SVC.SVC(X_train, y_train, C=C, iter=iter)
        model.train()
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        if acc > best_score:
            best_score = acc
            best_model = model
    return best_model

# ------------------------- Run Experiment ------------------------
results = []
svc_model = grid_search_svc(X_train, y_train, X_test, y_test)
y_pred_svc = svc_model.predict(X_test)
metrics_svc = get_metrics(y_test, y_pred_svc)
sparsity_svc = svc_model.support_vector_ratio()

for eta in eta_values:
    # Train PDProx
    pd_model = grid_search_pdprox(X_train, y_train, X_test, y_test, eta)
    y_pred_pd = pd_model.predict(X_test)
    metrics_pd = get_metrics(y_test, y_pred_pd)
    sparsity_pd = np.sum(pd_model.alpha > 1e-3) / len(pd_model.alpha)

    # Save results
    results.append({
        'eta': eta,
        'PDProx Accuracy': metrics_pd['accuracy'],
        'PDProx F1': metrics_pd['f1'],
        'PDProx Sparsity (α)': sparsity_pd,
        'SVC Accuracy': metrics_svc['accuracy'],
        'SVC F1': metrics_svc['f1'],
        'SVC Sparsity (SVs)': sparsity_svc
    })

# ------------------------- Display Results ------------------------
df = pd.DataFrame(results)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200) 
print(df.round(4))
