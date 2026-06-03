from scipy.stats import chi2, norm, nbinom, beta, uniform, poisson
import numpy as np


def probabilidades_funcion_discreta(f, expresiones, dominio, *args):
    probs = []
    for expresion in expresiones:

        if isinstance(expresion, int):
            if expresion<dominio[0] or expresion>dominio[1]:
                probs.append(0)
            else:
                probs.append(f(expresion, *args))

        else:
            a, b = expresion
            if a is None: 
                n = np.arange(dominio[0], b+1)
                probs.append(np.sum(f(n,*args))) # P(X <= b)
            elif b is None: 
                n = np.arange(a, dominio[1]+1)
                probs.append(np.sum(f(n,*args))) # P(X >= a)
            else: 
                n = np.arange(a, b+1)
                probs.append(np.sum(f(n,*args))) # P(a <= X <= b)

    return probs


def probabilidades_discretas(dist, expresiones):
    probs = []
    for expresion in expresiones:
        
        if isinstance(expresion, int):
            probs.append(dist.pmf(expresion)) # P(X = k)

        else:
            a, b = expresion
            if a is None: probs.append(dist.cdf(b)) # P(X <= b)
            elif b is None: probs.append(dist.sf(a - 1)) # P(X >= a)
            else: probs.append(dist.cdf(b) - dist.cdf(a - 1)) # P(a <= X <= b)

    return probs


def probabilidades_continuas(dist, expresiones):
    probs = []

    for expresion in expresiones:
        a, b = expresion
        if a is None: probs.append(dist.cdf(b)) # P(X < b)

        elif b is None: probs.append(dist.sf(a)) # P(X > a) v P(a < X)

        else: probs.append(dist.cdf(b) - dist.cdf(a)) # P(a < X < b)

    return probs


def agrupar_esp(observados, esperados, minimo=5):
    obs, esp = [], []
    fo, fe = 0, 0

    for i in range(len(observados)):
        fo += observados[i]
        fe += esperados[i]

        if fe >= minimo:
            obs.append(fo)
            esp.append(fe)
            fo, fe = 0, 0

    if fe > 0:
        if len(obs) > 0:
            obs[-1] += fo
            esp[-1] += fe
        else:
            obs.append(fo)
            esp.append(fe)

    return obs, esp


def chi_obs_bondad_ajuste(observados, probs):
    n = sum(observados)
    esperados = [n*probs[i] for i in range(len(observados))]
    observados, esperados = agrupar_esp(observados, esperados)

    chi_observado = sum((fo-fe)*(fo-fe)/fe for fo, fe in zip(observados, esperados))
    gl = len(observados) - 1
    print(f'Observados: {observados}')
    print(f'Probabilidades: {probs}')
    print(f'Esperados: {esperados}')
    print(f'Grados de libertad: {gl}')
    print(f'Chi Cuadrado observado: {chi_observado}')
    return chi_observado, gl, observados, esperados


def f_obs_anova(grupos):
    k = len(grupos)
    n = [len(grupo) for grupo in grupos]
    t = [sum(grupo) for grupo in grupos]
    N = sum(n)
    T = sum(t)
    datos = [x for grupo in grupos for x in grupo]
    suma_cuadrados = sum(x*x for x in datos)
    SST = suma_cuadrados - T*T/N
    SSA = sum([t[i]*t[i]/n[i] for i in range(k)]) - T*T/N
    SSE = SST - SSA
    f_obs = (SSA/SSE)*(N-k)/(k-1)
    gl = (k-1, N-k)
    print(f'k = {k}\n'
          f'n_i = {n}\n'
          f'T_i = {t}\n'
          f'N = {N}\n'
          f'T = {T}\n'
          f'ΣΣ(y_i_j)^2 = {suma_cuadrados}\n'
          f'SST = {SST}\n'
          f'SSA = {SSA}\n'
          f'SSE = {SSE}\n'
          f'Grados de libertad = {gl}\n'
          f'F_obs = {f_obs}')

    return f_obs, gl

def chi_obs_independencia(tabla_obs):
    fil = len(tabla_obs)
    col = len(tabla_obs[0])

    total_cols = [sum(tabla_obs[i][j] for i in range(fil)) for j in range(col)]
    total_fils = [sum(fila) for fila in tabla_obs]
    n = sum(total_fils)
    tabla_esp = [[total_fils[i]*total_cols[j]/n for j in range(col)] for i in range(fil)]
    chi_obs = sum(((fo-fe)**2)/fe for fila_obs, fila_esp in zip(tabla_obs, tabla_esp) for fo, fe in zip(fila_obs, fila_esp))
    gl = (fil-1)*(col-1)
    print(f'Total de cada fila: {total_fils}\n' 
          f'Total de cada columna: {total_cols}\n' 
          f'Total de fila × columna: {n}'
    )
    print("Esperados")
    for fila in tabla_esp: print(fila)

    print(f'Chi Cuadrado observado: {chi_obs}\n'
          f'Grados de libertad = {gl}'
    )
    return chi_obs, gl


import numpy as np

def regresion_lineal_simple(X, Y, alpha=None, beta=None):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    n = len(X)
    suma_x = np.sum(X)
    suma_y = np.sum(Y)
    suma_cuadrados_x = np.sum(X ** 2)
    suma_cuadrados_y = np.sum(Y ** 2)
    suma_x_y = np.sum(X * Y)
    Sxx = suma_cuadrados_x - (suma_x ** 2) / n
    Syy = suma_cuadrados_y - (suma_y ** 2) / n
    Sxy = suma_x_y - (suma_x * suma_y) / n

    if Syy == 0:
        r = 0
    else:
        r = Sxy / np.sqrt(Sxx * Syy)

    if beta is None and alpha is not None: beta = (suma_x_y - alpha * suma_x) / suma_cuadrados_x
    elif beta is not None and alpha is None: alpha = (suma_y - beta * suma_x) / n
    elif beta is None and alpha is None: beta, alpha = np.polyfit(X, Y, 1)

    print(f'Cantidad de datos, n = {n}\n'
      f'X = {X}\n'
      f'Y = {Y}\n'
      f'Σx = {suma_x}\n'
      f'Σy = {suma_y}\n'
      f'Σx^2 = {suma_cuadrados_x}\n'
      f'Σy^2 = {suma_cuadrados_y}\n'
      f'Σxy = {suma_x_y}\n'
      f'Sxx = {Sxx}\n'
      f'Syy = {Syy}\n'
      f'Sxy = {Sxy}\n'
      f'α = {alpha}\n'
      f'β = {beta}\n'
      f'r = {r}\n'
      f'R² = {r ** 2}')

    return lambda x: alpha + beta * x, alpha, beta, r

