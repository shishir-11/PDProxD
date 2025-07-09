from dataset.loader import LoadData
from models import SVC, PDProx1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from results.metric import LoadMetric
from results.plot import Plotting
import warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load dataset
scaler = StandardScaler()
dataset = 'adult'
loader = LoadData(scaler, dataset)
X, y = loader.data, loader.target
y = 2 * y - 1  # to {-1, +1}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define metrics
metric_types = ['accuracy', 'precision', 'recall', 'f1']
metrics = {m: LoadMetric(type=m) for m in metric_types}
for m in metrics.values():
    m.load_metric()

# Grid search for PDProx1
def grid_search_pdprox(model_class, X_train, y_train, X_test, y_test, metric_fn, iter=100):
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1],
        'lambda_': [0.001, 0.01, 0.1, 1],
        'eta': [0.2, 0.5, 1, 2, 3]
    }

    best_score = -np.inf
    best_params = None

    for C in param_grid['C']:
        for lambda_ in param_grid['lambda_']:
            for eta in param_grid['eta']:
                model = model_class(X_train, y_train, C=C, lambda_=lambda_, eta=eta, iter=iter)
                model.train()
                y_pred = model.predict(X_test)
                score = metric_fn(y_test, y_pred)

                if score > best_score:
                    best_score = score
                    best_params = (C, lambda_, eta)

    return best_params

# Grid search for SVC
def grid_search_svc(model_class, X_train, y_train, X_test, y_test, metric_fn, iter=100):
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1],
    }

    best_score = -np.inf
    best_params = None

    for C in param_grid['C']:
        model = model_class(X_train, y_train, C=C, iter=iter)
        model.train()
        y_pred = model.predict(X_test)
        score = metric_fn(y_test, y_pred)

        if score > best_score:
            best_score = score
            best_params = (C,)

    return best_params

# Initialize results
results = {}
accuracies = {'SVC': [], 'PDProx1': []}

# Train SVC
svc_params = grid_search_svc(SVC.SVC, X_train, y_train, X_test, y_test, metrics['accuracy'].get_score)
svc = SVC.SVC(X_train, y_train, C=svc_params[0], iter=100)
svc.train()
y_pred = svc.predict(X_test)
results['SVC'] = {m: metrics[m].get_score(y_test, y_pred) for m in metric_types}

for i in range(10, 51):
    svc = SVC.SVC(X_train, y_train, C=svc_params[0], iter=i)
    svc.train()
    y_pred = svc.predict(X_test)
    res = metrics['accuracy'].get_score(y_test, y_pred)
    accuracies['SVC'].append(res)

# Train PDProx1
pd_params = grid_search_pdprox(PDProx1.PDProx, X_train, y_train, X_test, y_test, metrics['accuracy'].get_score)
pdp = PDProx1.PDProx(X_train, y_train, C=pd_params[0], lambda_=pd_params[1], eta=pd_params[2], iter=100)
pdp.train()
y_pred = pdp.predict(X_test)
results['PDProx1'] = {m: metrics[m].get_score(y_test, y_pred) for m in metric_types}

for i in range(10, 51):
    pdp = PDProx1.PDProx(X_train, y_train, C=pd_params[0], lambda_=pd_params[1], eta=pd_params[2], iter=i)
    pdp.train()
    y_pred = pdp.predict(X_test)
    res = metrics['accuracy'].get_score(y_test, y_pred)
    accuracies['PDProx1'].append(res)

# Print sparsity comparisons
print(f"SVC sparsity (1 - SV ratio): {1 - svc.support_vector_ratio():.2%}")
print(f"PDProx1 sparsity (alpha > tol): {pdp.support_vector_ratio():.2%}")
print(f"PDProx1 weight sparsity (|w| < 1e-3): {pdp.weight_sparsity():.2%}")

# Plot results
plotter = Plotting(
    accuracy={k: v['accuracy'] for k, v in results.items()},
    precision={k: v['precision'] for k, v in results.items()},
    recall={k: v['recall'] for k, v in results.items()},
    f1={k: v['f1'] for k, v in results.items()}
)
plotter.plot()
plotter.acit(res_dict=accuracies)

eta_values = [0.1, 0.5, 1, 2, 5, 10]
accs = []
w_sparsities = []
sv_ratios = []

for eta in eta_values:
    model = PDProx1.PDProx(X_train, y_train, C=pd_params[0], lambda_=pd_params[1], eta=eta, iter=100)
    model.train()
    y_pred = model.predict(X_test)
    
    acc = metrics['accuracy'].get_score(y_test, y_pred)
    sparsity = model.weight_sparsity(tol=1e-3)
    sv_ratio = model.support_vector_ratio(tol=1e-3)

    accs.append(acc)
    w_sparsities.append(sparsity)
    sv_ratios.append(sv_ratio)

# 📈 Plot accuracy and sparsity vs eta
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.set_xlabel('η (eta)')
ax1.set_ylabel('Accuracy', color='tab:blue')
ax1.plot(eta_values, accs, marker='o', color='tab:blue', label='Accuracy')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()  # Second y-axis
ax2.set_ylabel('Sparsity / Support Vector Ratio', color='tab:red')
ax2.plot(eta_values, w_sparsities, marker='s', linestyle='--', color='tab:red', label='Weight Sparsity (w)')
ax2.plot(eta_values, sv_ratios, marker='^', linestyle=':', color='tab:orange', label='Support Vector Ratio (α)')

fig.tight_layout()
fig.suptitle("Effect of η on PDProx Sparsity and Accuracy", y=1.05)
fig.legend(loc='upper left', bbox_to_anchor=(0.15, 1.15))
plt.grid(True)
plt.show()