# part1_core_pytorch/ch05_mechanics/01_manual_gradient.py
"""
Chapter 5 Sections 5.2-5.4: Manual Gradient Descent
As per the text: "We'll first work things out by hand and then learn how to use PyTorch's super-powers."
"""
import torch

# 5.2.4 Choosing a linear model
def model(t_u, w, b):
    return w * t_u + b

# 5.3 Less loss is what we want (MSE)
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()

# 5.4.2 Getting analytical (Derivatives)
def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = 2 * (t_p - t_c) / t_p.size(0)
    dloss_dw = dloss_dtp * t_u
    dloss_db = dloss_dtp * 1.0
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])

# Normalization functions
def normalize_minmax(data):
    """Min-Max Normalization: scales data to [0, 1]"""
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

def normalize_zscore(data):
    """Z-Score Normalization: centers data at mean with std=1"""
    mean = data.mean()
    std = data.std()
    return (data - mean) / std



def main():
    # 5.2.2 Gathering data (Thermometer Calibration)
    t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
    t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
    
    t_un = 0.1 * t_u  # 5.4.1 Scaling inputs (Feature Scaling)
    
    # 5.4.3 Iterating to fit the model
    params = torch.tensor([1.0, 0.0])   # [w, b]
    learning_rate = 1e-2
    n_epochs = 5000

    print("Starting Manual Gradient Descent...")
    for epoch in range(n_epochs):
        w, b = params
        t_p = model(t_un, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_un, t_c, t_p, w, b)
        
        params = params - learning_rate * grad
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}, Params {params}")

    print(f"\nFinal Params: {params}")
    print("Expected (approx): [5.55, -17.77] (Fahrenheit to Celsius conversion)")

if __name__ == "__main__":
    main()