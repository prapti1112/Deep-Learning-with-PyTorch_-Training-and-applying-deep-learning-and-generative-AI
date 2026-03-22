# part1_core_pytorch/ch05_mechanics/03_optimizers.py
"""
Chapter 5 Section 5.5.2: Optimizers à la carte
As per the text: "The torch module has an optim submodule where we can find classes implementing different optimization algorithms."
"""
import torch
import torch.optim as optim

def main():
    # Data
    t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
    t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
    t_un = 0.1 * t_u

    def model(t_u, w, b):
        return w * t_u + b

    def loss_fn(t_p, t_c):
        squared_diffs = (t_p - t_c) ** 2
        return squared_diffs.mean()

    # 5.5.2 Using a Gradient Descent Optimizer
    params = torch.tensor([1.0, 0.0], requires_grad=True)
    optimizer = optim.SGD([params], lr=1e-2)
    n_epochs = 5000

    print("Starting Optimizer Training...")
    for epoch in range(n_epochs):
        # 5.5.2 Zeroing gradients is critical
        optimizer.zero_grad()
        
        t_p = model(t_un, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()
        
        # 5.5.2 Optimizer handles the update step
        optimizer.step()
            
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")

    print(f"\nFinal Params: {params.detach()}")
    print("Expected (approx): [5.55, -17.77] (Fahrenheit to Celsius conversion)")


if __name__ == "__main__":
    main()