from scipy.stats import t
from regresion_univariada import regresion_lineal_simple
import numpy as np

def calcular_estadisticos(X, Y, alpha=None, beta=None):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    modelo, alpha, beta, Sxx, Syy, Sxy, r, s_cuad, gl = regresion_lineal_simple(
        X, Y, alpha=alpha, beta=beta
    )

    n = len(X)

    return {
        'n': n,
        'X': X,
        'Y': Y,
        'alpha': alpha,
        'beta': beta,
        'Sxx': Sxx,
        'Syy': Syy,
        'Sxy': Sxy,
        'r': r,
        'r_cuadrado': r ** 2,
        'media_x': np.mean(X),
        'media_y': np.mean(Y),
        'gl': gl,
        'varianza_error': s_cuad,
        'error_estandar': np.sqrt(s_cuad),
        'modelo': modelo
    }


def obtener_t(confianza, gl):
    delta = 1 - confianza
    return t.ppf(1 - delta / 2, df=gl)


def imprimir_intervalo(titulo, confianza, valor_t, etiqueta_intervalo, limite_inferior, limite_superior):
    print(f"\n{'-'*30}")
    print(f"{titulo} {confianza*100:.0f}%")
    print(f"{'-'*30}")
    print(f"t(δ/2, n-2) = {valor_t:.6f}")
    print(f"{etiqueta_intervalo}: ]{limite_inferior:.6f}, {limite_superior:.6f}[")


def ic_parametro_beta(X, Y, alpha=None, beta=None, confianza=0.95):
    est = calcular_estadisticos(X, Y, alpha, beta)

    beta = est['beta']
    valor_t = obtener_t(confianza, est['gl'])

    error_estandar_beta = est['error_estandar'] * np.sqrt(1 / est['Sxx'])
    margen = valor_t * error_estandar_beta

    limite_inferior = beta - margen
    limite_superior = beta + margen

    imprimir_intervalo(
        titulo="IC para β",
        confianza=confianza,
        valor_t=valor_t,
        etiqueta_intervalo="IC",
        limite_inferior=limite_inferior,
        limite_superior=limite_superior
    )

    return limite_inferior, limite_superior


def ic_parametro_alpha(X, Y, alpha=None, beta=None, confianza=0.95):
    est = calcular_estadisticos(X, Y, alpha, beta)

    alpha = est['alpha']
    valor_t = obtener_t(confianza, est['gl'])

    suma_cuadrados_x = np.sum(est['X'] ** 2)

    error_estandar_alpha = est['error_estandar'] * np.sqrt(
        suma_cuadrados_x / (est['n'] * est['Sxx'])
    )

    margen = valor_t * error_estandar_alpha

    limite_inferior = alpha - margen
    limite_superior = alpha + margen

    imprimir_intervalo(
        titulo="IC para α",
        confianza=confianza,
        valor_t=valor_t,
        etiqueta_intervalo="IC",
        limite_inferior=limite_inferior,
        limite_superior=limite_superior
    )

    return limite_inferior, limite_superior


def ic_media_condicional(X, Y, x0, alpha=None, beta=None, confianza=0.95):
    est = calcular_estadisticos(X, Y, alpha, beta)

    alpha = est['alpha']
    beta = est['beta']
    valor_t = obtener_t(confianza, est['gl'])

    prediccion_puntual = alpha + beta * x0

    factor_variabilidad = (
        1 / est['n'] +
        (x0 - est['media_x']) ** 2 / est['Sxx']
    )

    error_estandar_media = est['error_estandar'] * np.sqrt(factor_variabilidad)
    margen = valor_t * error_estandar_media

    limite_inferior = prediccion_puntual - margen
    limite_superior = prediccion_puntual + margen

    imprimir_intervalo(
        titulo=f"IC para E(Y | x0 = {x0})",
        confianza=confianza,
        valor_t=valor_t,
        etiqueta_intervalo="IC",
        limite_inferior=limite_inferior,
        limite_superior=limite_superior
    )

    return limite_inferior, limite_superior


def ip_prediccion_individual(X, Y, x0, alpha=None, beta=None, confianza=0.95):
    est = calcular_estadisticos(X, Y, alpha, beta)

    alpha = est['alpha']
    beta = est['beta']
    valor_t = obtener_t(confianza, est['gl'])

    prediccion_puntual = alpha + beta * x0

    factor_variabilidad = (
        1 +
        1 / est['n'] +
        (x0 - est['media_x']) ** 2 / est['Sxx']
    )

    error_estandar_prediccion = est['error_estandar'] * np.sqrt(factor_variabilidad)
    margen = valor_t * error_estandar_prediccion

    limite_inferior = prediccion_puntual - margen
    limite_superior = prediccion_puntual + margen

    imprimir_intervalo(
        titulo=f"IP para y0 individual cuando x0 = {x0}",
        confianza=confianza,
        valor_t=valor_t,
        etiqueta_intervalo="IP",
        limite_inferior=limite_inferior,
        limite_superior=limite_superior
    )

    return limite_inferior, limite_superior
