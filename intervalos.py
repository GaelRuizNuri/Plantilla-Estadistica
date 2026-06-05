from scipy.stats import t
import numpy as np


def calcular_estadisticos(X, Y, alpha, beta):
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

    grados_libertad = n - 2
    suma_cuadrados_error = Syy - beta * Sxy
    varianza_error = suma_cuadrados_error / grados_libertad #s²
    error_estandar = np.sqrt(varianza_error) #s

    return {
        'n': n,
        'suma_x': suma_x,
        'suma_y': suma_y,
        'suma_cuadrados_x': suma_cuadrados_x,
        'suma_cuadrados_y': suma_cuadrados_y,
        'suma_x_y': suma_x_y,
        'Sxx': Sxx,
        'Syy': Syy,
        'Sxy': Sxy,
        'media_x': suma_x / n,
        'media_y': suma_y / n,
        'grados_libertad': grados_libertad,
        'varianza_error': varianza_error,
        'error_estandar': error_estandar,
    }


def ic_parametro_beta(X, Y, alpha, beta, confianza=0.95):
    est = calcular_estadisticos(X, Y, alpha, beta)
    delta = 1 - confianza
    valor_t = t.ppf(1 - delta / 2, df=est['grados_libertad'])
    error_estandar_beta = est['error_estandar'] * np.sqrt(1 / est['Sxx'])
    margen = valor_t * error_estandar_beta
    limite_inferior = beta - margen
    limite_superior = beta + margen

    print(f"\n{'='*45}")
    print(f"IC {confianza*100:.0f}% para β")
    print(f"{'='*45}")
    print(f"s² = {est['varianza_error']:.6f}")
    print(f"s = {est['error_estandar']:.6f}")
    print(f"g.l. = {est['grados_libertad']}")
    print(f"t(δ/2, n-2) = {valor_t:.6f}")
    #print(f"Error estándar de = {error_estandar_beta:.6f}")
    #print(f"Margen = {margen:.6f}")
    print(f"β estimado (b) = {beta:.6f}")
    print(f"IC: ]{limite_inferior:.6f}, {limite_superior:.6f}[")

    return limite_inferior, limite_superior


def ic_parametro_alpha(X, Y, alpha, beta, confianza=0.95):
    est = calcular_estadisticos(X, Y, alpha, beta)
    delta = 1 - confianza
    valor_t = t.ppf(1 - delta / 2, df=est['grados_libertad'])
    error_estandar_alpha = est['error_estandar'] * np.sqrt(est['suma_cuadrados_x'] / (est['n'] * est['Sxx']))
    margen = valor_t * error_estandar_alpha
    limite_inferior = alpha - margen
    limite_superior = alpha + margen

    print(f"\n{'='*45}")
    print(f"IC {confianza*100:.0f}% para α (intercepto)")
    print(f"{'='*45}")
    print(f"s² = {est['varianza_error']:.6f}")
    print(f"s = {est['error_estandar']:.6f}")
    print(f"g.l. = {est['grados_libertad']}")
    print(f"t(δ/2, n-2) = {valor_t:.6f}")
    #print(f"Error estándar de α = {error_estandar_alpha:.6f}")
    #print(f"Margen = {margen:.6f}")
    print(f"α estimado (a) = {alpha:.6f}")
    print(f"IC: ]{limite_inferior:.6f}, {limite_superior:.6f}[")

    return limite_inferior, limite_superior


def ic_media_condicional(X, Y, alpha, beta, x0, confianza=0.95):
    est = calcular_estadisticos(X, Y, alpha, beta)
    delta = 1 - confianza
    valor_t = t.ppf(1 - delta / 2, df=est['grados_libertad'])
    prediccion_puntual = alpha + beta * x0
    factor_variabilidad = 1/est['n'] + (x0 - est['media_x'])**2 / est['Sxx']
    error_estandar_media = est['error_estandar'] * np.sqrt(factor_variabilidad)
    margen = valor_t * error_estandar_media
    limite_inferior = prediccion_puntual - margen
    limite_superior = prediccion_puntual + margen

    print(f"\n{'='*45}")
    print(f"IC {confianza*100:.0f}% para E(Y | x0 = {x0})")
    print(f"{'='*45}")
    print(f"s² = {est['varianza_error']:.6f}")
    print(f"s = {est['error_estandar']:.6f}")
    print(f"g.l. = {est['grados_libertad']}")
    print(f"t(δ/2, n-2) = {valor_t:.6f}")
    print(f"x̄ = {est['media_x']:.6f}")
    print(f"ŷ0 = a + b·x0 = {prediccion_puntual:.6f}")
    print(f"Factor (1/n+(x0-x̄)²/Sxx) = {factor_variabilidad:.6f}")
    #print(f"Error estándar de ŷ0 = {error_estandar_media:.6f}")
    #print(f"Margen = {margen:.6f}")
    print(f"IC: ]{limite_inferior:.6f}, {limite_superior:.6f}[")

    return limite_inferior, limite_superior


def ip_prediccion_individual(X, Y, alpha, beta, x0, confianza=0.95):
    est = calcular_estadisticos(X, Y, alpha, beta)
    delta = 1 - confianza
    valor_t = t.ppf(1 - delta / 2, df=est['grados_libertad'])
    prediccion_puntual = alpha + beta * x0
    factor_variabilidad = 1 + 1/est['n'] + (x0 - est['media_x'])**2 / est['Sxx']
    error_estandar_prediccion = est['error_estandar'] * np.sqrt(factor_variabilidad)
    margen = valor_t * error_estandar_prediccion
    limite_inferior = prediccion_puntual - margen
    limite_superior = prediccion_puntual + margen

    print(f"\n{'='*45}")
    print(f"IP {confianza*100:.0f}% para y0 individual (x0 = {x0})")
    print(f"{'='*45}")
    print(f"s² = {est['varianza_error']:.6f}")
    print(f"s = {est['error_estandar']:.6f}")
    print(f"g.l. = {est['grados_libertad']}")
    print(f"t(δ/2, n-2) = {valor_t:.6f}")
    print(f"x̄ = {est['media_x']:.6f}")
    print(f"ŷ0 = a + b·x0 = {prediccion_puntual:.6f}")
    print(f"Factor (1+1/n+(x0-x̄)²/Sxx) = {factor_variabilidad:.6f}")
    #print(f"Error estándar de y0 = {error_estandar_prediccion:.6f}")
    #print(f"Margen = {margen:.6f}")
    print(f"IP: ]{limite_inferior:.6f}, {limite_superior:.6f}[")

    return limite_inferior, limite_superior
