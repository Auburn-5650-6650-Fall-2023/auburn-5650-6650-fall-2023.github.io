# Before you start

This repository hosts the course material for Math 5650/6650: **Nonlinear Optimization**. The class will use the textbook written by Jorge Nocedal and Stephen Wright, *Numerical Optimization*, 2nd Edition.

This course involves both basic optimization theory and programming. The prerequisites for the theory part are

- Linear Differential Equations (2650)
- Topics in Linear Algebra (2660)

```{note}
The default programming language for this class is ``Python``, the other script languages such as ``MATLAB``,  ``R``, ``Julia`` are also supported. 
```

## Formulation of Optimization Problems

The optimization serves as an important part of data science. Optimization algorithms are widely used in machine learning, statistics, and other fields.

An optimization problem can be formulated as follows:

- an objective function: $f(x)$, which is meant to be maximized or minimized
- decision variables: $x \in \mathbb{R}^n$
- constraints: $g_i(x) \le 0$, $h_j(x) = 0$

The standard form of an optimization problem is

$$\min_{x \in \mathbb{R}^n} f(x),\quad \text{subject to } \begin{cases}g_i(x) \le 0, \quad i = 1, \ldots, m\\h_j(x) = 0, \quad j = 1, \ldots, p\end{cases}$$

````{prf:definition}
:label: def-feasible-solution
A **feasible solution** is a solution that satisfies all the constraints, otherwise, the solution is called **infeasible**.

The **feasible region** of an optimization problem is the set of all feasible solutions. The feasible region is also called the **feasible set**. 
````

````{prf:example}
:label: ex-feasible-region
Consider the following optimization problem:

$$\min_{x \in \mathbb{R}^2} f(x) = (x_1-2)^2 + (x_2-1)^2, \quad \text{subject to } \begin{cases}x_2 - x_1^2 \ge 0\\x_1 + x_2 \le 2\end{cases}$$

We can reformulate the optimization problem with the standard form:

- the objective function: $f(x) = (x_1-2)^2 + (x_2-1)^2$
- decision variables: $x = (x_1, x_2)$
- constraints: $g_1(x) = x_2 - x_1^2 \ge 0$, $g_2(x) = x_1 + x_2 \le 2$

````

The following code snippet shows the feasible region of the optimization problem.

```{code-cell} ipython3
:tags: [scroll-output]
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

<!-- 
````{tab-set}
```{tab-item} Tab 1 title
My first tab
```

```{tab-item} Tab 2 title
My second tab with `some code`!
```
````

::::{grid}
:gutter: 2

:::{grid-item}
:outline:
A
:::
:::{grid-item}
:outline:
B
:::
:::{grid-item}
:outline:
C
:::
:::{grid-item}
:outline:
D
:::

:::: -->