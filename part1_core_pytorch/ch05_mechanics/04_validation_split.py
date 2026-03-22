# part1_core_pytorch/ch05_mechanics/04_validation_split.py
"""
Chapter 5 Section 5.5.3: Training, Validation, and Overfitting
As per the text: "If the training loss and the validation loss diverge, we're overfitting."
"""
import torch
import torch.optim as optim

def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()

def main():
    # Full Data
    t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
    t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
    t_un = 0.1 * t_u

    # 5.5.3 Splitting a Dataset
    n_samples = t_u.shape[0]
    n_val = int(0.2 * n_samples)
    shuffled_indices = torch.randperm(n_samples)
    train_indices = shuffled_indices[:-n_val]
    val_indices = shuffled_indices[-n_val:]

    train_t_u = t_un[train_indices]
    train_t_c = t_c[train_indices]
    val_t_u = t_un[val_indices]
    val_t_c = t_c[val_indices]

    params = torch.tensor([1.0, 0.0], requires_grad=True)
    optimizer = optim.SGD([params], lr=1e-2)
    n_epochs = 3000

    print("Starting Training with Validation Split...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Training Pass
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)
        train_loss.backward()
        optimizer.step()
        
        # 5.5.5 Switching off autograd for validation
        with torch.no_grad():
            val_t_p = model(val_t_u, *params)
            val_loss = loss_fn(val_t_p, val_t_c)
            
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Train Loss {train_loss.item():.4f}, Val Loss {val_loss.item():.4f}")

    print(f"\nFinal Params: {params.detach()}")
    print("Expected (approx): [5.55, -17.77] (Fahrenheit to Celsius conversion)")


if __name__ == "__main__":
    main()