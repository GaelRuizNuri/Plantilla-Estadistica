import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def regresion_lineal_simple(f, X, Y, alpha, beta, r, inicio=None, fin=None, cantidad=100, show = False, file_name = None):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    if inicio is None:
        inicio = np.min(X)
    if fin is None:
        fin = np.max(X)

    x_grafica = np.linspace(inicio, fin, cantidad)
    y_grafica = f(x_grafica)

    plt.figure(figsize=(8, 5))
    plt.plot(x_grafica, y_grafica, color="red", label="Recta de regresión")
    plt.scatter(X, Y, label="Puntos dados")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"y = {alpha:.4f} + {beta:.4f}x\nR² = {r**2:.4f}")
    plt.legend()
    plt.grid(True)

    if (file_name is not None): plt.savefig(file_name, dpi=300, bbox_inches="tight")
    if (show): plt.show()

def bondad_ajuste(observados, esperados, etiquetas=None, show = False, file_name = None):
    observados = np.array(observados, dtype=float)
    esperados = np.array(esperados, dtype=float)

    if etiquetas is None:
        etiquetas = [str(i + 1) for i in range(len(observados))]

    x = np.arange(len(etiquetas))
    ancho = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - ancho/2, observados, width=ancho, label="Observados")
    plt.bar(x + ancho/2, esperados, width=ancho, label="Esperados")

    plt.xticks(x, etiquetas)
    plt.xlabel("Categorías")
    plt.ylabel("Frecuencia")
    plt.title("Frecuencias observadas vs esperadas")
    plt.legend()
    plt.grid(axis="y")

    if (file_name is not None): plt.savefig(file_name, dpi=300, bbox_inches="tight")
    if (show): plt.show()

def distribucion_discreta(dist, inicio, fin, titulo="Distribución discreta", show = False, file_name = None):
    x = np.arange(inicio, fin + 1)
    y = dist.pmf(x)
    plt.figure(figsize=(8, 5))
    plt.bar(x, y)
    plt.xlabel("x")
    plt.ylabel("P(X = x)")
    plt.title(titulo)
    plt.grid(axis="y")
    if (file_name is not None): plt.savefig(file_name, dpi=300, bbox_inches="tight")
    if (show): plt.show()

def distribucion_continua(dist, inicio, fin, cantidad=300, titulo="Distribución continua", show = False, file_name = None):
    x = np.linspace(inicio, fin, cantidad)
    y = dist.pdf(x)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color="red")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(titulo)
    plt.grid(True)
    if (file_name is not None): plt.savefig(file_name, dpi=300, bbox_inches="tight")
    if (show): plt.show()

