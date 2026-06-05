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
