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


def r2_original(Y, Y_pred):
    Y = np.array(Y, dtype=float)
    Y_pred = np.array(Y_pred, dtype=float)

    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)

    return 1 - ss_res / ss_tot


def metricas_error(Y, Y_pred):
    Y = np.array(Y, dtype=float)
    Y_pred = np.array(Y_pred, dtype=float)

    residuos = Y - Y_pred

    mae = np.mean(np.abs(residuos))
    mse = np.mean(residuos ** 2)
    rmse = np.sqrt(mse)

    return mae, mse, rmse


def regresion_no_lineal_linealizable(X, Y, tipo):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    modelo_lineal = None
    tipo = tipo.lower()

    if tipo == "exponencial":
        # y = a * b^x
        # ln(y) = ln(a) + xln(b)

        if np.any(Y <= 0):
            raise ValueError("El modelo exponencial requiere que todos los valores de Y sean positivos.")

        X_transformado = X
        Y_transformado = np.log(Y)

        modelo_lineal, alpha_lineal, beta_lineal, Sxx, Syy, Sxy, r, s_cuad, gl = regresion_lineal_simple(X_transformado, Y_transformado)

        a = np.exp(alpha_lineal)
        b = np.exp(beta_lineal)

        def modelo_original(x):
            x = np.array(x, dtype=float)
            return a * pow(b,x);

        formula = f"y = {a:.6f} * {b:.6f}^x"

    elif tipo == "potencial":
        # y = a * x^b
        # ln(y) = ln(a) + b ln(x)

        if np.any(X <= 0) or np.any(Y <= 0):
            raise ValueError("El modelo potencial requiere que X y Y sean positivos.")

        X_transformado = np.log(X)
        Y_transformado = np.log(Y)

        modelo_lineal, alpha_lineal, beta_lineal, Sxx, Syy, Sxy, r, s_cuad, gl = regresion_lineal_simple(
            X_transformado,
            Y_transformado
        )

        a = np.exp(alpha_lineal)
        b = beta_lineal

        def modelo_original(x):
            x = np.array(x, dtype=float)
            return a * (x ** b)

        formula = f"y = {a:.6f} * x^({b:.6f})"

    elif tipo == "logaritmico":
        # y = a + b ln(x)

        if np.any(X <= 0):
            raise ValueError("El modelo logarítmico requiere que todos los valores de X sean positivos.")

        X_transformado = np.log(X)
        Y_transformado = Y

        modelo_lineal, alpha_lineal, beta_lineal, Sxx, Syy, Sxy, r, s_cuad, gl = regresion_lineal_simple(X_transformado, Y_transformado)

        a = alpha_lineal
        b = beta_lineal

        def modelo_original(x):
            x = np.array(x, dtype=float)
            return a + b * np.log(x)

        formula = f"y = {a:.6f} + {b:.6f} ln(x)"

    elif tipo == "inverso":
        # y = a + b / x
        # y = a + b * (1/x)

        if np.any(X == 0):
            raise ValueError("El modelo inverso no permite valores de X iguales a 0.")

        X_transformado = 1 / X
        Y_transformado = Y

        modelo_lineal, alpha_lineal, beta_lineal, Sxx, Syy, Sxy, r, s_cuad, gl = regresion_lineal_simple(X_transformado, Y_transformado)

        a = alpha_lineal
        b = beta_lineal

        def modelo_original(x):
            x = np.array(x, dtype=float)
            return a + b / x

        formula = f"y = {a:.6f} + {b:.6f}/x"


    Y_pred = modelo_original(X)

    r2 = r2_original(Y, Y_pred)
    mae, mse, rmse = metricas_error(Y, Y_pred)

    print("\n" + "=" * 50)
    print(f"Modelo no lineal: {tipo}")
    print("=" * 50)
    print(f"Fórmula ajustada: {formula}")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"R² en escala original = {r2}")
    print(f"MAE = {mae}")
    print(f"MSE = {mse}")
    print(f"RMSE = {rmse}")

    return {
        "tipo": tipo,
        "modelo": modelo_original,
        "modelo_lineal" : modelo_lineal,
        "formula" : formula,
        "a": a,
        "b": b,
        "alpha_lineal": alpha_lineal,
        "beta_lineal": beta_lineal,
        "X_transformado": X_transformado,
        "Y_transformado": Y_transformado,
        "Sxx": Sxx,
        "Syy": Syy,
        "Sxy": Sxy,
        "r_transformado": r,
        "r2_transformado": r ** 2,
        "r2_original": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "s_cuad": s_cuad,
        "gl": gl
    }


def comparar_modelos_no_lineales(X, Y, modelos=None):
    if modelos is None:
        modelos = ["exponencial", "potencial", "logaritmico", "inverso"]

    resultados = []

    for tipo in modelos:
        try:
            resultado = regresion_no_lineal_linealizable(X, Y, tipo)

            resultados.append({
                "tipo": tipo,
                "formula": resultado["formula"],
                "R2_original": resultado["r2_original"],
                "MAE": resultado["MAE"],
                "MSE": resultado["MSE"],
                "RMSE": resultado["RMSE"],
                "resultado": resultado
            })

        except ValueError as e:
            print(f"\nNo se pudo ajustar el modelo {tipo}: {e}")

    resultados_ordenados = sorted(resultados, key=lambda r: r["RMSE"])

    print("\n" + "=" * 70)
    print("Comparación de modelos")
    print("=" * 70)

    for r in resultados_ordenados:
        print(f"\nModelo: {r['tipo']}")
        print(f"Fórmula: {r['formula']}")
        print(f"R² original: {r['R2_original']:.6f}")
        print(f"RMSE: {r['RMSE']:.6f}")
        print(f"MAE: {r['MAE']:.6f}")

    if len(resultados_ordenados) > 0:
        mejor = resultados_ordenados[0]
        print("\n" + "-" * 70)
        print(f"Mejor modelo según RMSE: {mejor['tipo']}")
        print(f"Fórmula: {mejor['formula']}")
        print("-" * 70)

    return resultados_ordenados
