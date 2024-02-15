---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Line Search Methods

The search direction $p_k$ is a key component of line search algorithms. The search direction is a descent direction if it satisfies the following condition:

$$\nabla f_k^T p_k < 0.$$

The most natural search direction is the negative gradient direction $-\nabla f(x_k)$. There are other popular search directions, for instance, Newton direction, which will be discussed later.

## Step Length

Each iteration of line search methods requires a step length $\alpha_k$ and a search direction $p_k$ to be computed, the update is

$$x_{k+1} = x_k + \alpha_k p_k.$$

Once the search direction $p_k$ is computed, the step length $\alpha_k$ is then computed to reduce the objective function $f$. Usually, $\alpha_k$ should be compromised between the reduction of $f$ and the computational cost of computing $\alpha_k$. Ideally, the best choice is the so-called ``exact line search'' which finds the optimal $\alpha_k$ that minimizes the following single-variable function $\phi(\cdot)$ by

$$\phi(\alpha) := f(x_k + \alpha p_k),\quad \alpha > 0.$$

```{code-cell} ipython3
:tags: [hide-output]

from scipy.optimize import minimize_scalar

"""
@description: sample code for exact line search along the direction of pk
@parameters : 
    @objFunc    : objective function  
    @xk         : starting point 
    @pk         : search direction
@returns    : step size
@note       : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html
"""
def exact_line_search_method(objFunc, xk, pk):
    def subproblem1D(alpha):
        return objFunc(xk + alpha * pk)
    res = minimize_scalar(subproblem1D) 
    return res.x
```

### Wolfe Conditions

In general, it is quite expensive to find the optimal $\alpha_k$ in each iteration. Therefore, we usually use some simple strategies to find a good $\alpha_k$, which leads to the *inexact line search*. The most popular one is the so-called **Wolfe Conditions**.

The **Wolfe Conditions** are two inequalities that the step length $\alpha_k$ should satisfy:

- **Sufficient decrease condition** (also called **Armijo condition**): $f(x_k + \alpha_k p_k) \leq f(x_k) + c_1 \alpha_k \nabla f_k^T p_k$ with $0 < c_1 < 1$.
- **Curvature condition**: $\nabla f(x_k + \alpha_k p_k)^T p_k \geq c_2 \nabla f_k^T p_k$ with $0 < c_1 < c_2 < 1$.

The step length satisfying the **Wolfe Conditions** is called a **Wolfe step**, which may not be close to the exact step length. In order to force the steo length to be close to the exact step length, we can use the **strong Wolfe Conditions**:

- **Sufficient decrease condition**: $f(x_k + \alpha_k p_k) \leq f(x_k) + c_1 \alpha_k \nabla f_k^T p_k$ with $0 < c_1 < 1$.
- **Curvature condition**: $|\nabla f(x_k + \alpha_k p_k)^T p_k| \leq c_2 |\nabla f_k^T p_k|$ with $0 < c_1 < c_2 < 1$.

The only difference is the derivative $\phi'(\alpha_k)$ in the curvature condition is replaced by its absolute value $|\phi'(\alpha_k)|$, which cannot be too positive.

````{prf:lemma} Existence of Wolfe Step
:label: lemma-existence-wolfe-step
Suppose that $f$ is a continuously differentiable function. Let $p_k$ be a descent direction of $f$ at $x_k$ and assume $f$ is bounded below on the line $x_k + \alpha p_k$ for $\alpha > 0$. Then the step length $\alpha_k$ satisfying the (strong) Wolfe Conditions exists.
````

````{prf:proof}
The proof is omitted here.
````

```{code-cell} ipython3
:tags: [hide-output]
from scipy.optimize import line_search

"""
@description: sample code for wolfe step line search along the direction of pk
@parameters : 
    @objFunc    : objective function  
    @objFunc_Grad: gradient of objective function
    @xk         : starting point 
    @pk         : search direction
    @c1         : parameter for Armijo condition
    @c2         : parameter for curvature condition
@returns    : step size
@note       : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html
"""

def wolfe_line_search_method(objFunc, objFunc_Grad, xk, pk, c1=1e-4, c2=0.9):
    return line_search(objFunc, objFunc_Grad, xk, pk, c1=c1, c2=c2)[0]
```

### Goldstein Conditions

Another popular choice of the inexact line search is the **Goldstein Conditions**:

$$f(x_k) + c \alpha_k \nabla f_k^T p_k \le f(x_k + \alpha_k p_k) \leq f(x_k) + (1 - c) \alpha_k \nabla f_k^T p_k$$

with $0 < c < \frac{1}{2}$. The second inequality is the **sufficient decrease condition** and the first inequality is the control of step length from being too short. In comparison with Wolfe condition, one disadvantage of Goldstein condition is that the first inequality of the condition might exclude all minimizers of $\phi(\alpha)$. However, usually it is not a fatal problem as long as the objective decreases in the direction of convergence. As a short conclusion, the Goldstein and Wolfe conditions have quite similar convergence theories. Compared to the Wolfe conditions, the Goldstein conditions are often used in Newton-type methods but are not well-suited for quasi-Newton methods that maintain a positive definite Hessian approximation.

### Backtracking Line Search

The **backtracking line search** is a simple strategy to find a step length $\alpha_k$ that satisfies the sufficient decrease condition. The algorithm is described as follows:

````{prf:algorithm} Backtracking Line Search
:label: alg-backtracking-line-search

Given $x_k$, $p_k$, $\alpha_0$, $\rho \in (0, 1)$, $c \in (0, 1)$.

1. Set $\alpha = \alpha_0$.
2. While $f(x_k + \alpha p_k) > f(x_k) + c \alpha \nabla f_k^T p_k$ do
    1. $\alpha = \rho \alpha$.
3. End While
````

This strategy for terminating a line search is well suited for Newton methods but is less appropriate for quasi-Newton and conjugate gradient methods.

## Convergence

The convergence of line search methods 

## Newton’s Method and Quasi-Newton Methods

## Stochastic Gradient Descent

## More Topics

### Momentum Methods

### Nesterov’s Accelerated Gradient

## Exercises

### Theory

### Programming
