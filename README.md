# RationalRiccatiSolver
An ODE solver which uses [SymPy](https://www.sympy.org) to give all rational solutions to Riccati Equations. 

## Examples
These are results on SymPy 1.7.1

``` python
In [1]: from sympy.riccati import *
   ...: f = Function('f')
   ...: x = Symbol('x')

In [2]: eq = f(x).diff(x) + f(x)**2 - (4*x**6 - 8*x**5 + 12*x**4 + 4*x**3 +  7*x**2 - 20*x + 4)/(4*x**4)

In [3]: eq
Out[3]: 
                      6      5       4      3      2
 2      d          4⋅x  - 8⋅x  + 12⋅x  + 4⋅x  + 7⋅x  - 20⋅x + 4
f (x) + ──(f(x)) - ────────────────────────────────────────────
        dx                                4
                                       4⋅x

In [4]: sols = find_riccati_sol(eq)

In [5]: sols[0]
Out[5]: 
     2⋅x          3    1
x + ────── - 1 - ─── + ──
     2           2⋅x    2
    x  - 1             x

In [6]: checkodesol(eq, sols) # Verify if the solution is correct
Out[6]: [(True, 0)]

In [7]: eq = f(x).diff(x) + f(x)**2 - (2*x + 1/x)*f(x) + x**2

In [8]: eq
Out[8]: 
 2   ⎛      1⎞         2      d
x  - ⎜2⋅x + ─⎟⋅f(x) + f (x) + ──(f(x))
     ⎝      x⎠                dx

In [9]: sols = list(map(lambda x: x.simplify(), find_riccati_sol(eq))) # Simplify all solutions

In [10]: sols
Out[10]: 
⎡         ⎛      2    ⎞  ⎤
⎢    2  x⋅⎝C₁ + x  + 2⎠   ⎥
⎢x + ─, ───────────────, x⎥
⎢    x            2       ⎥
⎣           C₁ + x        ⎦

In [11]: checkodesol(eq, sols)
Out[11]: [(True, 0), (True, 0), (True, 0)]
```