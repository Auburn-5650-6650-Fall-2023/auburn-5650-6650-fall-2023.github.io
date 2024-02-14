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

# Introduction of Optimization

The optimization serves as an important tool for decision making, resource allocation, and many other applications. Optimization algorithms are widely used in machine learning, statistics, and other fields.

## Formulation of Optimization Problems

An optimization problem can be formulated as follows:

- an objective function: $f(x)$, which is meant to be maximized or minimized
- decision variables: $x \in \mathbb{R}^n$
- constraints: $g_i(x) \le 0$, $h_j(x) = 0$

The standard form of an optimization problem is

$$\min_{x \in \mathbb{R}^n} f(x),\quad \text{subject to } \begin{cases}g_i(x) \le 0, \quad i = 1, \ldots, m\\h_j(x) = 0, \quad j = 1, \ldots, p\end{cases}$$

The objective is to find the optimal decision variable $x^*$ that minimizes the objective function $f(x)$ while satisfying all the constraints.

````{prf:definition}
:label: def-feasible-solution
A **feasible solution** is a solution that satisfies all the constraints, otherwise, the solution is called **infeasible**.

The **feasible region** of an optimization problem is the set of all feasible solutions. The feasible region is also called the **feasible set**. 
````

In the following, we provide an example to illustrate the concept of the feasible region.

````{prf:example}
:label: ex-feasible-region

$$\min_{x \in \mathbb{R}^2} f(x) = (x_1-2)^2 + (x_2-1)^2, \quad \text{subject to } \begin{cases}x_2 - x_1^2 \ge 0\\x_1 + x_2 \le 2\end{cases}$$

We can reformulate the optimization problem with the standard form:

- the objective function: $f(x) = (x_1-2)^2 + (x_2-1)^2$
- decision variables: $x = (x_1, x_2)$
- constraints: $g_1(x) = x_2 - x_1^2 \ge 0$, $g_2(x) = x_1 + x_2 \le 2$

````

The following code snippet shows the feasible region of the optimization problem.

```{code-cell} ipython3
:tags: []
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

# Construct the feasible region
x = np.linspace(-3, 3, 2000)

y1 = x**2
y2 = 2 - x

plt.plot(x, y1, label=r'$x_2 - x_1^2\geq 0$')
plt.plot(x, y2, label=r'$2 - x_1 -x_2\geq 0$')
plt.xlim((-3, 3))
plt.ylim((0, 6))
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.fill_between(x, y2, y1, where=y2>y1, color='grey', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
```

### Continuous vs. Discrete Optimization

Some optimization problems have special requirements on the decision variables. For example, the decision variables are required to be integers or discrete values, then the optimization problem is called **discrete optimization**. Otherwise, the optimization problem is called **continuous optimization**.

The discrete optimization problems are usually more difficult to solve than continuous optimization problems. The discrete optimization problems are widely used in combinatorial optimization, such as the traveling salesman problem, the knapsack problem, and the assignment problem.

### Unconstrained vs. Constrained Optimization

Optimization problems can be classified into two categories: **unconstrained optimization** and **constrained optimization**.

- **Unconstrained optimization**: the optimization problem does not have any constraints or it is safe to ignore the constraints. Sometimes constrained optimization problems can be converted into unconstrained optimization problems by using the Lagrange multiplier method.

- **Constrained optimization**: the optimization problem has constraints, for instance, $g_i(x) \le 0$, $h_j(x) = 0$. The constraints can be linear or nonlinear, equality or inequality.

### Global vs. Local Optimization

Most fast optimization algorithms can only find a local minimum, which is not necessarily the global minimum. The global optimization problem is usually more difficult to solve than the local optimization problem. The global optimization problem is also called the **global optimization**.

One special case of the global optimization problem is the **convex optimization** problem, whose local minimums are also global minimums. The convex optimization problem is usually easier to solve than the general optimization problem.

## Optimization Algorithms

Most optimization algorithms are **iterative algorithms**, which means they start from an initial point $x_0\in\mathbb{R}^n$ and then generate a sequence of points $x_k$, $k=1,2,\cdots$ that converge to the optimal solution. The sequence of points is called the **iterative sequence**.

Ideally, a perfect algorithm should have the following properties:

- Robustness: the algorithm should be able to handle different types of optimization problems, such as convex, non-convex, smooth, non-smooth, etc.
- Efficiency: the algorithm should be able to find the optimal solution within a reasonable amount of time.
- Accuracy: the algorithm should be able to find the optimal solution with high precision.

These properties are usually conflicting with each other. For example, a rapidly convergent algorithm for nonlinear programming may require too much computation resources on large problems. On the other hand, a robust algorithm may also be the slowest one. The trade-off between these properties is a key issue in the design of optimization algorithms.

### Convexity

The **convexity** plays an important role in optimization. Usually it implies some benign properties of the optimization problem. The **convexity** applies to both sets and functions. 

- For sets, a set $C\subseteq\mathbb{R}^n$ is called **convex** if the line segment between any two points in $C$ is also in $C$. Mathematically, it means 
  
  $$\lambda x + (1-\lambda)y\in C, \quad \forall x, y\in C,\quad \lambda\in[0,1].$$

- For functions, a function $f(x)$ is called **convex** if its domain is a convex set and the following inequality holds
  
  $$f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y), \quad \forall x, y\in\text{dom}f,\quad \lambda\in[0,1].$$
