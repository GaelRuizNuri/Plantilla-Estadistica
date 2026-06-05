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
    r = 0 if Sxx == 0 or Syy == 0 else Sxy / np.sqrt(Sxx * Syy)
    if beta is None and alpha is not None: beta = (suma_x_y - alpha * suma_x) / suma_cuadrados_x
    elif beta is not None and alpha is None: alpha = (suma_y - beta * suma_x) / n
    elif beta is None and alpha is None: beta, alpha = np.polyfit(X, Y, 1)
    s_cuad = (Syy - beta * Sxy) / (n - 2) 
    gl = n - 2
    print(f'n = {n}\n'
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
          f's^2 = {s_cuad}\n'
          f'gl = {gl}\n'
          f'R² = {r ** 2}')

    return lambda x: alpha + beta * x, alpha, beta, Sxx, Syy, Sxy, r, s_cuad, gl
