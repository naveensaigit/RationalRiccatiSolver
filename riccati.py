from sympy import *
from sympy.solvers.ode.ode import get_numbered_constants, _remove_redundant_solutions, constantsimp

def find_riccati_sol(eq, log=False):
    """
    Finds and returns all rational solutions to a Riccati Differential Equation.
    """

    # Step 0 :Match the equation
    w = list(eq.atoms(Derivative))[0].args[0]
    x = list(w.free_symbols)[0]
    eq = eq.expand().collect(w)
    cf = eq.coeff(w.diff(x))
    eq = Add(*map(lambda x: cancel(x/cf), eq.args)).collect(w)
    b0, b1, b2 = match_riccati(eq, w, x)

    # Step 1 : Convert to Normal Form
    a = -b0*b2 + b1**2/4 - b1.diff(x)/2 + 3*b2.diff(x)**2/(4*b2**2) + b1*b2.diff(x)/(2*b2) - b2.diff(x, 2)/(2*b2)
    a_t = cancel(a.together())

    # Step 2
    presol = []

    # Step 3 : "a" is 0
    if a_t == 0:
        presol.append(1/(x + get_numbered_constants(eq)))

    # Step 4 : "a" is a non-zero constant
    elif a_t.is_complex:
        presol.append([sqrt(a), -sqrt(a)])

    # Step 5 : Find poles and valuation at infinity
    poles = find_poles(a_t, x)
    poles, muls = list(poles.keys()), list(poles.values())
    val_inf = val_at_inf(a_t, x)

    if log:
        print("Simplified Equation", eq)
        print("b0, b1, b2", b0, b1, b2)
        print("a", a)
        print("a_t", a_t)
        print("Constant Solutions", presol)
        print("Poles, Muls, val_inf", poles, muls, val_inf)

    if len(poles) and b2 != 0:
        # Check necessary conditions
        if val_inf%2 != 0 or not all(map(lambda mul: (mul == 1 or (mul%2 == 0 and mul >= 2)), muls)):
            raise ValueError("Rational Solution doesn't exist")

        # Step 6
        # Construct c-vectors for each singular point
        c = construct_c(a, x, poles, muls)

        # Construct d vectors for each singular point
        d = construct_d(a, x, val_inf)
        if log:
            print("C", c)
            print("D", d)

        # Step 7 : Iterate over all possible combinations and return solutions
        for it in range(2**(len(poles) + 1)):
            choice = list(map(lambda x: int(x), bin(it)[2:].zfill(len(poles) + 1)))
            # Step 8 and 9 : Compute m and ybar
            m, ybar = compute_degree(x, poles, choice, c, d, -val_inf//2)
            if log:
                print("M", m)
                print("Ybar", ybar)
            # Step 10 : Check if m is non-negative integer
            if m.is_real and m >= 0 and m.is_integer:
                # Step 11 : Find polynomial solutions of degree m for the auxiliary equation
                psol, coeffs, exists = solve_aux_eq(a, x, m, ybar)
                if log:
                    print("Psol, coeffs, exists", psol, coeffs, exists)
                # Step 12 : If valid polynomial solution exists, append solution.
                if exists:
                    if psol == 1 and coeffs == 0:
                        presol.append(ybar)
                    elif len(coeffs):
                        psol = psol.subs(coeffs[0])
                        presol.append(ybar + psol.diff(x)/psol)
    # Step 15 : Transform the solutions of the equation in normal form
    sol = list(map(lambda y: -y/b2 - b2.diff(x)/(2*b2**2) - b1/(2*b2), presol))
    return sol

def test_riccati_sol():
    f = Function('f')
    x, C1 = symbols('x C1')

    # Example from Fritz Schwarz - Algorithmic Lie Theory for Solving Ordinary Differential Equations
    # Pg 21, Example 2.5

    eq = f(x).diff(x) + f(x)**2 - (2*x + 1/x)*f(x) + x**2
    sols = list(map(lambda x: x.simplify(), find_riccati_sol(eq)))
    assert sols == [x + 2/x, x*(C1 + x**2 + 2)/(C1 + x**2), x]
    assert checkodesol(eq, sols) == [(True, 0), (True, 0), (True, 0)]

    # Example from Kovacic's Paper - https://www.sciencedirect.com/science/article/pii/S0747717186800104
    # Pg 13, Section 3.2, Example 1

    eq = f(x).diff(x) + f(x)**2 - (4*x**6 - 8*x**5 + 12*x**4 + 4*x**3 + 7*x**2 - 20*x + 4)/(4*x**4)
    sols = find_riccati_sol(eq)
    assert sols == [x + 2*x/(x**2 - 1) - 1 - 3/(2*x) + x**(-2)]
    assert checkodesol(eq, sols) == [(True, 0)]

    eq = f(x).diff(x) - 2*f(x)/x - x*f(x)**2
    sols = find_riccati_sol(eq)
    assert sols == [-4/x**2, (-4*x**3/(C1 + x**4) + 3/(2*x))/x - 3/(2*x**2), 0]
    assert checkodesol(eq, sols) == [(True, 0), (True, 0), (True, 0)]

    eq = f(x).diff(x) - f(x)**2 + (15*x**2 - 20*x + 7)/((2*x - 1)**2*(x - 1)**2)
    sols = find_riccati_sol(eq)
    sols = list(map(lambda x: x.simplify(), sols))
    assert sols == [(-15*x**2 + 15*x - 4)/(6*x**3 - 11*x**2 + 6*x - 1), (-60*x**3 + 180*x**2 - 181*x + 62)/(24*x**4 - 92*x**3 + 130*x**2 - 79*x + 17),
    (9*C1*x - 6*C1 - 15*x**5 + 60*x**4 - 94*x**3 + 72*x**2 - 30*x + 6)/(6*C1*x**2 - 9*C1*x + 3*C1 + 6*x**6 - 29*x**5 + 57*x**4 - 58*x**3 + 30*x**2 - 6*x),
    (3*x - 2)/((x - 1)*(2*x - 1))]
    assert checkodesol(eq, sols) == [(True, 0), (True, 0), (True, 0), (True, 0)]

    # Example 4.2 - Banks, Gundersen, Laine: Meromorphic Solutions of the Riccati Differential Equation
    # (http://www.acadsci.fi/mathematica/Vol06/vol06pp369-398.pdf)

    eq = f(x).diff(x) - f(x)**2 + 9*x**2/4 - S(21)/2
    sols = find_riccati_sol(eq)
    assert apart(sols[0]) == 3*x/2 - 1/(x + 1) - 1/(x - 1) - 1/x
    assert checkodesol(eq, sols) == [(True, 0)]

    # Example 4.4.8 - Rational and Algebraic Solutions of First-Order Algebraic ODEs
    # (https://www3.risc.jku.at/publications/download/risc_5387/PhDThesisThieu.pdf)

    # Takes too long!
    # eq = f(x).diff(x) + (3*x**2 + 1)*f(x)**2/x + (6*x**2 - x + 3)*f(x)/(x*(x - 1)) + (3*x**2 - 2*x + 2)/(x*(x - 1)**2)
    # sols = list(map(lambda x: x.simplify(), find_riccati_sol(eq)))
    # assert sols == [(-27*x**5 + 27*x**4 + 18*sqrt(3)*I*x**4 - 45*x**3 - 18*sqrt(3)*I*x**3 - 9*x**2 +
    # 30*sqrt(3)*I*x**2 + 6*x + 2*sqrt(3)*I)/(27*x**6 - 27*x**5 - 18*sqrt(3)*I*x**5
    # + 18*x**4 + 18*sqrt(3)*I*x**4 - 18*x**3 - 12*sqrt(3)*I*x**3 + 3*x**2 +
    # 12*sqrt(3)*I*x**2 - 3*x - 2*sqrt(3)*I*x + 2*sqrt(3)*I), (-27*x**5 + 27*x**4 -
    # 18*sqrt(3)*I*x**4 - 45*x**3 + 18*sqrt(3)*I*x**3 - 9*x**2 - 30*sqrt(3)*I*x**2
    # + 6*x - 2*sqrt(3)*I)/(27*x**6 - 27*x**5 + 18*sqrt(3)*I*x**5 + 18*x**4 -
    # 18*sqrt(3)*I*x**4 - 18*x**3 + 12*sqrt(3)*I*x**3 + 3*x**2 - 12*sqrt(3)*I*x**2 -
    # 3*x + 2*sqrt(3)*I*x - 2*sqrt(3)*I), (-C1 - x**3 + x**2 - 2*x)/(C1*x - C1 + x**4
    # - x**3 + x**2 - x), -1/(x - 1)]
    # assert checkodesol(eq, sols) == [(True, 0), (True, 0), (True, 0), (True, 0)]

    # Examples from https://www3.risc.jku.at/publications/download/risc_5197/RISCReport15-19.pdf

    # 1.14
    eq = f(x).diff(x) + f(x)**2 - 15/(4*x**2)
    sols = find_riccati_sol(eq)
    assert sols == [5/(2*x), 4*x**3/(C1 + x**4) - 3/(2*x), -3/(2*x)]
    assert checkodesol(eq, sols) == [(True, 0), (True, 0), (True, 0)]

    # 1.24
    eq = f(x).diff(x) + 3*f(x)**2 - 2/x**2
    sols = list(map(lambda x: x.expand(), find_riccati_sol(eq)))
    assert sols == [1/x, 5*x**4/(3*C1 + 3*x**5) - S(2)/(3*x), -S(2)/(3*x)]
    assert checkodesol(eq, sols) == [(True, 0), (True, 0), (True, 0)]

    # 1.26 - Assume A/a = B/b = k
    k = Symbol('k', integer=True, positive=True)
    a, b = symbols('a b')
    eq = f(x).diff(x) - a*b*k**2*f(x)**2 + 2*a*b*k*f(x) - a*b
    sols = find_riccati_sol(eq)
    assert sols == [1/k - 1/(a*b*k**2*(C1 + x))]
    assert checkodesol(eq, sols) == [(True, 0)]

    # 1.31 - Assume k = +/-8
    eq = f(x).diff(x) - 4*I*(1 + f(x)**2)/(2*x)
    sols = find_riccati_sol(eq.expand())
    assert sols == [I, -I*x*(-4*x**3/(C1 + x**4) + 3/(2*x))/2 - I/4, -I]
    assert checkodesol(eq, sols) == [(True, 0), (True, 0), (True, 0)]

    # 1.95
    eq = f(x).diff(x) - f(x)**2/x + 1/x
    sols = find_riccati_sol(eq)
    assert sols == [-1, x*(-2*x/(C1 + x**2) + 1/(2*x)) + S(1)/2, 1]
    assert checkodesol(eq, sols) == [(True, 0), (True, 0), (True, 0)]

    # 1.98 - Working only for n=1
    a, b, c = symbols('a b c')
    eq = c*x**(2*b) + f(x).diff(x)*x + a*f(x)**2 - b*f(x)
    sols = find_riccati_sol(eq.subs({a: -S(1)/2, b: 0, c: S(1)/2}))
    assert sols[0].simplify() == (C1 - x)/(C1 + x)
    assert checkodesol(eq.subs({a: -S(1)/2, b: 0, c: S(1)/2}), sols) == [(True, 0)]

    # 1.99 (a)
    eq = f(x).diff(x) + a*f(x)**2/x - b*f(x)/x - c/x
    sols = find_riccati_sol(eq.subs({a: S(3)/4, b: 1, c: 1}))
    assert sols == [2, -4*x*(-2*x/(C1 + x**2) + S(1)/(2*x))/3, -S(2)/3]
    assert checkodesol(eq.subs({a: S(3)/4, b: 1, c: 1}), sols) == [(True, 0), (True, 0), (True, 0)]

    # 1.99 (b)
    eq = f(x).diff(x) + a*f(x)**2/x - b*f(x)/x - c/x
    sols = find_riccati_sol(eq.subs({a: 1, b: 3, c: 4}))
    assert sols == [4, -x*(-5*x**4/(C1 + x**5) + 2/x) + 1, -1]
    assert checkodesol(eq.subs({a: 1, b: 3, c: 4}), sols) == [(True, 0), (True, 0), (True, 0)]

    # 1.101
    eq = f(x).diff(x) + f(x)**2 - f(x)/x
    sols = find_riccati_sol(eq)
    assert sols == [2/x, 2*x/(C1 + x**2), 0]
    assert checkodesol(eq, sols) == [(True, 0), (True, 0), (True, 0)]

    # 1.103
    eq = f(x).diff(x) - f(x)**2 - f(x)*(1/x + 2*x) - x**2
    sols = list(map(lambda x: x.simplify(), find_riccati_sol(eq)))
    assert sols == [-x - 2/x, -x*(C1 + x**2 + 2)/(C1 + x**2), -x]
    assert checkodesol(eq, sols) == [(True, 0), (True, 0), (True, 0)]

    # 1.107
    alpha, beta = symbols('alpha beta')
    eq = f(x).diff(x) + b*f(x)/x + a*f(x)**2*x**(alpha-1) - c*x**(beta-1)
    sols = list(map(lambda x: x.expand(), find_riccati_sol(eq.subs({a: -S(3)/4, c: 5, alpha: 4, beta: -4, b: 2}))))
    assert sols == [-4/(3*x**4) - 2*sqrt(11)*I/(3*x**4), -4/(3*x**4) + 2*sqrt(11)*I/(3*x**4)]
    assert checkodesol(eq.subs({a: -S(3)/4, c: 5, alpha: 4, beta: -4, b: 2}), sols) == [(True, 0), (True, 0)]

    # 1.139 - Equivalent to condition 1 of 1.14

    # 1.140
    eq = f(x).diff(x) + f(x)**2 + 4*f(x)/x + 2/x**2
    sols = find_riccati_sol(eq)
    assert sols == [1/(C1 + x) - 2/x]
    assert checkodesol(eq, sols) == [(True, 0)]

    # 1.143
    eq = f(x).diff(x) + a*f(x)**2 - b/x**2
    sols = list(map(lambda x: x.expand(), find_riccati_sol(eq.subs({a: S(5)/4, b: 7}))))
    assert sols == [S(14)/(5*x), 24*x**5/(5*C1 + 5*x**6) - 2/x, -2/x]
    assert checkodesol(eq.subs({a: S(5)/4, b: 7}), sols) == [(True, 0), (True, 0), (True, 0)]

    # 1.144 - Note: Piecewise condition is wrong in the paper, must be (1 - n**2)/4 (Compare with 1.143).
    eq = f(x).diff(x) + a*f(x)**2 + (b + c)/x**2
    sols = list(map(lambda x: x.expand(), find_riccati_sol(eq.subs({a: -S(11)/2, b: 2, c: S(5)/2}))))
    assert sols == [-1/x, -20*x**9/(11*C1 + 11*x**10) + S(9)/(11*x), 9/(11*x)]
    assert checkodesol(eq.subs({a: -S(11)/2, b: 2, c: S(5)/2}), sols) == [(True, 0), (True, 0), (True, 0)]

    # 1.165
    eq = 2*f(x).diff(x)*x**2 - (4*f(x) + f(x).diff(x) - 4)*x + (f(x) - 1)*f(x)
    sols = find_riccati_sol(eq)
    assert sols[0].simplify() == (C1 + 2*x**2)/(C1 + x)
    assert checkodesol(eq, sols) == [(True, 0)]

    # 1.171
    eq = f(x).diff(x)*x**3 - f(x)**2 - x**2*f(x)
    sols = find_riccati_sol(eq)
    assert sols[0].simplify() == C1*x**2/(C1 + x)
    assert checkodesol(eq, sols) == [(True, 0)]

    # 1.172
    eq = -f(x)**2*x**4 + f(x).diff(x)*x**3 + f(x)*x**2 + 20
    sols = find_riccati_sol(eq)
    assert sols == [-5/x**2, (-9*x**8/(C1 + x**9) + 4/x)/x, 4/x**2]
    assert checkodesol(eq, sols) == [(True, 0), (True, 0), (True, 0)]

    # 1.177
    eq = f(x).diff(x)*x**3 - (f(x) + f(x).diff(x))*x**2 + 2*f(x)*x - f(x)**2
    sols = find_riccati_sol(eq)
    assert sols[0].simplify() == x**2*(C1 + 1)/(C1 + x)
    assert checkodesol(eq, sols) == [(True, 0)]

    # 1.182
    eq = f(x).diff(x)*x**4 + x**2 - (2*f(x)**2 + f(x).diff(x))*x + f(x)
    sols = list(map(lambda x: x.simplify(), find_riccati_sol(eq)))
    assert sols == [1/x, x*(C1*x + 1)/(C1 + x**2), x**2]
    assert checkodesol(eq, sols) == [(True, 0), (True, 0), (True, 0)]

def solve_aux_eq(a, x, m, ybar):
    p = Function('p')
    auxeq = numer((p(x).diff(x, 2) + 2*ybar*p(x).diff(x) + (ybar.diff(x) + ybar**2 - a)*p(x)).together())
    psyms = get_numbered_constants(auxeq, m)
    if type(psyms) != tuple:
        psyms = (psyms, )
    psol = x**m
    for i in range(m):
        psol += psyms[i]*x**i
    if m != 0:
        return psol, solve(auxeq.subs(p(x), psol).doit().expand(), psyms, dict=True), True
    else:
        cf = auxeq.subs(p(x), psol).doit().expand()
        return S(1), cf, cf == 0

def compute_degree(x, poles, choice, c, d, N):
    ybar = 0
    m = sympify(d[choice[0]][-1])

    for i in range(len(poles)):
        for j in range(len(c[i][choice[i + 1]])):
            ybar += c[i][choice[i + 1]][j]/(x - poles[i])**(j+1)
        m -= c[i][choice[i + 1]][0]
    for i in range(N+1):
        ybar += d[choice[0]][i]*x**i
    return m, ybar

def construct_d(a, x, val_inf):
    ser = a.series(x, oo)
    N = -val_inf//2
    # Case 4
    if val_inf < 0:
        temp = [0 for i in range(N+2)]
        temp[N] = sqrt(ser.coeff(x, 2*N))
        for s in range(N-1, -2, -1):
            sm = 0
            for j in range(s+1, N):
                sm += temp[j]*temp[N+s-j]
            if s != -1:
                temp[s] = (ser.coeff(x, N+s) - sm)/(2*temp[N])
        temp1 = list(map(lambda x: -x, temp))
        temp[-1] = (ser.coeff(x, N+s) - temp[N] - sm)/(2*temp[N])
        temp1[-1] = (ser.coeff(x, N+s) - temp1[N] - sm)/(2*temp1[N])
        d = [temp, temp1]
    # Case 5
    elif val_inf == 0:
        temp = [0, 0]
        temp[0] = sqrt(ser.coeff(x, 0))
        temp[-1] = ser.coeff(x, -1)/(2*temp[0])
        d = [temp, list(map(lambda x: -x, temp))]
    # Case 6
    else:
        s_inf = limit(x**2*a, x, oo)
        d = [[(1 + sqrt(1 + 4*s_inf))/2], [(1 - sqrt(1 + 4*s_inf))/2]]
    return d

def construct_c(a, x, poles, muls):
    c = []
    for pole, mul in zip(poles, muls):
        c.append([])
        # Case 3
        if mul == 1:
            c[-1].extend([[1], [1]])
        # Case 1
        elif mul == 2:
            r = a.series(x, pole).coeff(x - pole, -2)
            c[-1].extend([[(1 + sqrt(1 + 4*r))/2], [(1 - sqrt(1 + 4*r))/2]])
        # Case 2
        else:
            ri = mul//2
            ser = a.series(x, pole)
            temp = [0 for i in range(ri)]
            temp[ri-1] = sqrt(ser.coeff(x - pole, -mul))
            for s in range(ri-1, 0, -1):
                sm = 0
                for j in range(s+1, ri):
                    sm += temp[j-1]*temp[ri+s-j-1]
                if s!= 1:
                    temp[s-1] = (ser.coeff(x - pole, -ri-s) - sm)/(2*temp[ri-1])
            temp1 = list(map(lambda x: -x, temp))
            temp[0] = (ser.coeff(x - pole, -ri-s) - sm + ri*temp[ri-1])/(2*temp[ri-1])
            temp1[0] = (ser.coeff(x - pole, -ri-s) - sm + ri*temp1[ri-1])/(2*temp1[ri-1])
            c[-1].extend([temp, temp1])
    return c

def match_riccati(eq, w, x):
    b0 = Wild('b0', exclude=[w, w.diff(x)])
    b1 = Wild('b1', exclude=[w, w.diff(x)])
    b2 = Wild('b2', exclude=[w, w.diff(x)])
    match = eq.match(w.diff(x) - b0 - b1*w - b2*w**2)

    if b0 not in match or b1 not in match or b2 not in match:
        raise ValueError("Invalid Riccati Equation")

    return match[b0], match[b1], match[b2]

def find_poles(a, x):
    p = roots(denom(a))
    a = cancel(a.subs(x, 1/x).together())
    p_inv = roots(denom(a))
    if 0 in p_inv:
        p.update({oo: p_inv[0]})
    return p

def val_at_inf(a, x):
    num, denom = a.as_numer_denom()
    return degree(denom, x) - degree(num, x)

def match_2nd_order(eq, z, x):
    a = Wild('a', exclude=[z, z.diff(x), z.diff(x)*2])
    b = Wild('b', exclude=[z, z.diff(x), z.diff(x)*2])
    c = Wild('c', exclude=[z, z.diff(x), z.diff(x)*2])
    match = eq.match(a*z.diff(x, 2) + b*z.diff(x) + c*z)

    if a not in match or b not in match or c not in match:
        raise ValueError("Invalid Second Order Linear Homogeneous Equation")
    return match[a], match[b], match[c]

def find_kovacic_simple(x, a, b, c):
    # Find solution for y(x).diff(x, 2) = r(x)*y(x)
    r = (b**2 + 2*a*b.diff(x) - 2*a.diff(x)*b - 4*a*c)/4*a**2

    fric = Function('fric')
    ric_sol = find_riccati_sol(fric(x).diff(x) + fric(x)**2 - r)

    C1 = Symbol('C1')
    return set(map(lambda sol: exp(Integral(sol.subs(C1, 0), x).doit()), ric_sol))
    
def find_kovacic_sol(eq):
    z = list(eq.atoms(Derivative))[0].args[0]
    x = list(z.free_symbols)[0]
    eq = eq.expand().collect(z)
    a, b, c = match_2nd_order(eq, z, x)

    # Transform the differential equation to a simpler form
    # using z(x) = y(x)*exp(Integral(-b/(2*a))) and find its solution
    ysol = find_kovacic_simple(x, a, b, c)

    zsol = list(map(lambda sol: sol*exp(Integral(-b/(2*a), x).doit()), ysol))
    zsol = _remove_redundant_solutions(Eq(eq, 0), list(map(lambda sol: Eq(z, sol), zsol)), 2, x)

    C1, C2 = symbols('C1 C2')
    if len(zsol) == 2:
        return constantsimp(Eq(z, C1*zsol[0].rhs + C2*zsol[1].rhs), [C1, C2])
    if len(zsol) == 1:
        sol1 = zsol[0].rhs
        sol2 = sol1*Integral(exp(Integral(-b, x).doit())/sol1**2, x)
        zsol.append(Eq(z, sol2))
        return constantsimp(Eq(z, C1*zsol[0].rhs + C2*zsol[1].rhs), [C1, C2])
    return zsol

def test_kovacic_sol():
    eq = f(x).diff(x, 2) - (x**2 + 3)*f(x)
    sol = find_kovacic_sol(eq)
    assert checkodesol(eq, sol)

    eq = f(x).diff(x, 2) - 2*f(x)/x**2
    sol = find_kovacic_sol(eq)
    assert checkodesol(eq, sol)

    eq = f(x).diff(x, 2) + (3/(16*x**2*(x - 1)**2))*f(x)
    sol = find_kovacic_sol(eq)
    assert checkodesol(eq, sol)

    # eq = f(x).diff(x, 2) - f(x)/(8*(25*x + 16)**2*(x - 2)**2)
    # sol = find_kovacic_sol(eq)
    # assert checkodesol(eq, sol)

    eq = f(x).diff(x, 2) - (9*x**2/4 - S(21)/2)*f(x)
    sol = find_kovacic_sol(eq)
    assert checkodesol(eq, sol)
    
    eq = f(x).diff(x, 2) + 3*x*f(x).diff(x) + 12*f(x)
    sol = find_kovacic_sol(eq)
    assert checkodesol(eq, sol)