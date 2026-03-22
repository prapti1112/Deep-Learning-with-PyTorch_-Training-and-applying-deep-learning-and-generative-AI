# Chapter 5: The Mechanics of Learning

## 🎯 Value Proposition
This chapter demystifies **learning** as **parameter estimation** via **gradient descent**. You will understand how loss functions guide parameter updates via gradients, moving from manual derivation to PyTorch's automatic differentiation (`autograd`).

## 📂 File Structure
| File | Section | Concept |
| :--- | :--- | :--- |
| `01_manual_gradient.py` | 5.2-5.4 | Manual gradient calculation (Numerical & Analytical) |
| `02_autograd.py` | 5.5.1 | Using `requires_grad` and `loss.backward()` |
| `03_optimizers.py` | 5.5.2 | Using `torch.optim.SGD` and `optimizer.step()` |
| `04_validation_split.py` | 5.5.3 | Training/Validation split to detect overfitting |

## 🔑 Key Concepts
1.  **Loss Function:** A score (e.g., MSE) that the learning process attempts to minimize.
2.  **Gradient Descent:** Updating parameters in the direction of decreasing loss (`params -= lr * grad`).
3.  **Autograd:** PyTorch's engine that tracks operations on tensors with `requires_grad=True` to compute gradients automatically.
4.  **Gradient Accumulation:** Gradients accumulate by default; you **must** call `optimizer.zero_grad()` or `param.grad.zero_()` every iteration.
5.  **Overfitting:** When training loss decreases but validation loss increases. Detected by splitting data into Train/Val sets.
6.  **Inference Mode:** Use `torch.no_grad()` during validation to save memory and compute.

## 📝 Exercises (Section 5.7)
1.  **Quadratic Model:** Redefine model to `w2 * t_u** 2 + w1 * t_u + b`.
    *   What parts of the training loop need to change?
    *   Is the resulting loss higher or lower?
2.  **(Implicit)** Experiment with learning rates to observe divergence (loss → `inf`).

## ⚠️ Common Pitfalls
*   **Forgetting `zero_grad()`:** Leads to exploding gradients.
*   **Updating params without `no_grad()`:** Unnecessary graph tracking during updates.
*   **Not Normalizing Inputs:** Can cause gradient imbalance between parameters (Section 5.4.4).
*   **Validating with Gradients On:** Wastes memory; risks accidental training on validation data.