"KAN net"
import torch
import torch.nn.functional as F
from utils.metrics_utils import root_mean_squared_error
from kan import KAN

# X_train_flat = X_train.reshape(X_train.shape[0], -1)
# X_test_flat  = X_test.reshape(X_test.shape[0], -1)
X_train_mean = X_train.mean(axis=1)
X_test_mean  = X_test.mean(axis=1)

X_train_t = torch.tensor(X_train_mean, dtype=torch.float32)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_mean, dtype=torch.float32)
y_test_t  = torch.tensor(y_test_scaled, dtype=torch.float32)

dataset = {
    'train_input': X_train_t,
    'train_label': y_train_t,
    'test_input': X_test_t,
    'test_label': y_test_t,
    'train_ratio': 0.8}

model = KAN(width=[X_train_t.shape[1], 128, y_train_t.shape[1]], grid=5, k=3)

print("Starting training...")
results = model.fit(
    dataset, 
    opt="LBFGS",
    steps=5,
    lamb=0.01)

print("\nTraining Complete.")
print(f"Final training loss: {results['train_loss'][-1]:.4e}")
print(f"Final test loss: {results['test_loss'][-1]:.4e}")
with torch.no_grad():
    y_pred_t = model(X_test_t)
y_pred = y_pred_t.cpu().numpy()

rmse = root_mean_squared_error(y_test_t.cpu().numpy(), y_pred)
print(f"Test RMSE: {rmse:.4f}")

X_train_mean = X_train.mean(axis=1)
X_test_mean  = X_test.mean(axis=1)

X_train_t = torch.tensor(X_train_mean, dtype=torch.float32)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_mean, dtype=torch.float32)
y_test_t  = torch.tensor(y_test_scaled, dtype=torch.float32)

model = KAN(width=[X_train_t.shape[1], 64, 32, y_train_t.shape[1]], grid=5, k=3)

results = model.fit(
    {'train_input': X_train_t, 'train_label': y_train_t,
    'test_input': X_test_t,  'test_label': y_test_t,
    'train_ratio': 0.8},
    opt="LBFGS", steps=30, lamb=0.01, update_grid=True)

with torch.no_grad():
    y_pred_t = model(X_test_t)
y_pred = y_pred_t.cpu().numpy()

rmse = root_mean_squared_error(y_test_t.cpu().numpy(), y_pred)
print(f"[KAN] Test RMSE: {rmse:.4f}")
model.plot()
