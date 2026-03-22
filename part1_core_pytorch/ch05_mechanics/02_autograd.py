# part1_core_pytorch/ch05_mechanics/02_autograd.py
"""
Chapter 5 Section 5.5.1: PyTorch Autograd
As per the text: "PyTorch tensors can remember where they come from... and automatically provide the chain of derivatives."
"""
import torch

def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()

def main():
    # Data (Normalized)
    t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
    t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
    t_un = 0.1 * t_u

    # 5.5.1 Applying Autograd
    params = torch.tensor([1.0, 0.0], requires_grad=True)
    learning_rate = 1e-2
    n_epochs = 5000

    print("Starting Autograd Training...")
    for epoch in range(n_epochs):
        # 5.5.1 Accumulating Gradients Gotcha
        if params.grad is not None:
            params.grad.zero_()
        
        t_p = model(t_un, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()
        
        # 5.5.1 Switching off autograd for updates
        with torch.no_grad():
            params -= learning_rate * params.grad
            
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")

    print(f"\nFinal Params: {params.detach()}")
    print("Expected (approx): [5.55, -17.77] (Fahrenheit to Celsius conversion)")

if __name__ == "__main__":
    main()