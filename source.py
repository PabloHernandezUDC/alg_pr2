# Iván Moure Pérez, Pablo Hernández Martínez
# i.moure@udc.es, pablo.hernandez.martinez@udc.es

import random, time
import numpy as np
from prettytable import PrettyTable

# EJERCICIO Nº1
'''
En este ejercicio, se definen algoritmos de ordenación y funciones auxiliares que se utilizarán posteriormente para analizar su rendimiento.
'''
print("\n\n***Ejercicio 1***\n")

# hibbard_increments(array_length): Genera una secuencia de incrementos de Hibbard hasta un tamaño dado.
def hibbard_increments(array_length):
    increments = []
    k = 1
    gap = 2**k - 1
    while gap < array_length:
        increments.insert(0, gap)
        k += 1
        gap = 2**k - 1
    return increments

# shell_sort_hibbard(v): Ordena un vector utilizando el algoritmo Shell con la secuencia de incrementos de Hibbard.
def shell_sort_hibbard(v):
    increments = hibbard_increments(len(v))
    return shell_sort_aux(v, increments)

# shell_sort_aux(v, increments): Ordena un vector utilizando el algoritmo de ordenación Shell con una secuencia de incrementos dada.
def shell_sort_aux(v, increments):
    for increment in increments:
        for i in range(increment, len(v)):
            tmp = v[i]
            j = i
            seguir = True
            while (j - increment) >= 0 and seguir:
                if tmp < v[j-increment]:
                    v[j] = v[j-increment]
                    j -= increment
                else:
                    seguir = False
            v[j] = tmp
    return v

# ins_sort(v): Ordena un vector utilizando el algoritmo de inserción.
def ins_sort(v):
    n = len(v)
    for i in range(1, n):
        x = v[i]
        j = i - 1
        while j >= 0 and v[j] > x:
            v[j+1] = v[j]
            j -= 1
        v[j+1] = x
    return v

# aleatorio(n): Genera un vector de números aleatorios de longitud n.
def aleatorio(n):
    v = list(range(n))
    for i in v:
        v[i] = random.randint(-n, n)
    return v

# calcular_tiempo(func, v, tipo): Mide el tiempo de ejecución de una función con un vector dado y un tipo de inicialización.
def calcular_tiempo(func, v, tipo):
    start = time.perf_counter_ns()
    func(v)
    finish = time.perf_counter_ns()
    t = finish - start
    # solo se ejecuta si no cumplimos con el umbral de confianza
    if t < 500000: # 500 microsegundos = 500000 ns
        n = len(v)
        k = 100
        
        start = time.perf_counter_ns()
        for i in range(k):
            vector = generar_vector(tipo, n)
            func(vector)
        finish = time.perf_counter_ns()
        t1 = finish - start

        start = time.perf_counter_ns()
        for i in range(k):
            vector = generar_vector(tipo, n)
        finish = time.perf_counter_ns()
        t2 = finish - start

        t = (t1 - t2) / k
        if t < 0:
            print(f'///// Valor negativo con n={n}, anomalía.')

    return t

# generar_vector(tipo, n): Genera un vector de un tipo específico (ascendente, descendente o aleatorio) de longitud n.
def generar_vector(tipo, n):
    
    # Vector tipo 1: ascendiente
    # Vector tipo 2: descendiente
    # Vector tipo 3: aleatorio
    
    if tipo in (1, 2):
        vector = np.arange(0, n)
        if tipo == 2:
            vector = np.flipud(vector)
    elif tipo == 3:
        vector = aleatorio(n)

    return vector

# test(): Realiza pruebas de ordenación con vectores aleatorios e imprime los resultados.
def test():
    n = 5
    vector = aleatorio(n)
    
    print(f'\nInicializacion aleatoria')
    print(vector)
    
    print('\nOrdenación por inserción')
    result = ins_sort(vector.copy())
    print(result)
    if result == sorted(vector):
        print('///Éxito.')
    else:
        print('---Fracaso.')

    print('\nInicializacion descendiente')
    vector = sorted(vector, reverse = True)
    print(vector)
    
    print('\nOrdenación Shell')
    result = shell_sort_hibbard(vector.copy())
    print(result)
    if result == sorted(vector):
        print('///Éxito.')
    else:
        print('---Fracaso.')
    
    print()

# EJERCICIO Nº2
'''
En este ejercicio, se ejecuta la función test() para realizar pruebas de ordenación con vectores aleatorios.
'''
print("\n\n***Ejercicio 2*** ")
test()

# EJERCICIO Nº3
'''
En este ejercicio, se realiza un análisis de rendimiento de los algoritmos de ordenación en diferentes configuraciones y se crean tablas
para registrar los resultados.
'''
print("\n\n***Ejercicio 3*** ")
# Creamos las 6 tablas pedidas
ascIns,  ascShell  = PrettyTable(), PrettyTable()
descIns, descShell = PrettyTable(), PrettyTable()
randIns, randShell = PrettyTable(), PrettyTable()

# Establecemos los nombres de los distintos parámetros de las tablas
ascIns.title          = 'Ordenacion por inserción con inicialización ascendiente'
ascIns.field_names    = ['n', 't(n) (ns)', 't(n)/n', ' t(n)/n**1.04', 't(n)/n**1.08']
ascShell.title        = 'Ordenacion Shell con inicialización ascendiente'
ascShell.field_names  = ['n', 't(n) (ns)', 't(n)/n', ' t(n)/n**1.2', 't(n)/n**1.4']
descIns.title         = 'Ordenacion por inserción con inicialización descendiente'
descIns.field_names   = ['n', 't(n) (ns)', 't(n)/n**1.8', ' t(n)/n**1.99', 't(n)/n**2.18']
descShell.title       = 'Ordenacion Shell con inicialización descendiente'
descShell.field_names = ['n', 't(n) (ns)', 't(n)/n', ' t(n)/n**1.11', 't(n)/n**1.22']
randIns.title         = 'Ordenacion por inserción con inicialización aleatoria'
randIns.field_names   = ['n', 't(n) (ns)', 't(n)/n**1.8', ' t(n)/n**2.0', 't(n)/n**2.2']
randShell.title       = 'Ordenacion Shell con inicialización aleatoria'
randShell.field_names = ['n', 't(n) (ns)', 't(n)/n', ' t(n)/n**1.09', 't(n)/n**1.18']

# Calculamos tiempo de ejecución (inicio)
totalStart = time.perf_counter_ns()

# A continuación, se realizan cálculos de tiempo y se llenan las tablas con los resultados...
n = 128
for i in range(8):
    vector = generar_vector(1, n)

    t = calcular_tiempo(ins_sort, vector, 1)
    ascIns.add_row([n, (str(t) + '*' if type(t) == float else t), t/n, t/n**1.04, t/n**1.08])

    t = calcular_tiempo(shell_sort_hibbard, vector, 1)
    ascShell.add_row([n, (str(t) + '*' if type(t) == float else t), t/n, t/n**1.2, t/n**1.4])

    n *= 2

n = 128
for i in range(8):
    vector = generar_vector(2, n)

    t = calcular_tiempo(ins_sort, vector, 2)
    descIns.add_row([n, (str(t) + '*' if type(t) == float else t), t/n**1.8, t/n**1.99, t/n**2.18])

    t = calcular_tiempo(shell_sort_hibbard, vector, 2)
    descShell.add_row([n, (str(t) + '*' if type(t) == float else t), t/n, t/n**1.11, t/n**1.22])

    n *= 2

n = 128 
for i in range(8):
    vector = generar_vector(3, n)

    t = calcular_tiempo(ins_sort, vector, 3)
    randIns.add_row([n, (str(t) + '*' if type(t) == float else t), t/n**1.8, t/n**2.0, t/n**2.2])

    t = calcular_tiempo(shell_sort_hibbard, vector, 3)
    randShell.add_row([n, (str(t) + '*' if type(t) == float else t), t/n, t/n**1.09, t/n**1.18])

    n *= 2

# Calculamos tiempo de ejecución (final)
totalFinish = time.perf_counter_ns()

# Aclaración sobre asteriscos
print('\nLos datos que se muestran acompañados de un asterisco (*) indican que los tiempos',
      'fueron medidos de nuevo ya que no cumplían con el umbral de confianza de 500 microsegundos.',
      'El bucle que calcula la media se iteró 100 veces.', sep = '\n')

# Mostramos las tablas (ya completas) con su correspondiente nombre
print()
print(ascIns)
print()

print(ascShell)
print()

print(descIns)
print()

print(descShell)
print()

print(randIns)
print()

print(randShell)
print()

# Mostramos el tiempo de ejecución
print(f'Tiempo total de ejecución: {round((totalFinish - totalStart) / (10**9), 2)}s.')
print()

# EJERCICIO Nº4
'''
En este ejercicio, se calcula empíricamente la complejidad de los algoritmos de ordenación y se ajustan los
resultados a la complejidad esperada.

print("\n\n***Ejercicio 4*** ")
# Función para calcular la complejidad empírica
def calcular_complejidad_empirica(algoritmo, situacion_inicial, n_min, n_max, paso):
    resultados = []
    for n in range(n_min, n_max + 1, paso):
        vector = generar_vector(situacion_inicial, n)
        tiempo = calcular_tiempo(algoritmo, vector, situacion_inicial)
        resultados.append((n, tiempo))
    return resultados

# Función para ajustar los resultados a una complejidad
def ajustar_complejidad(resultados, complejidad_esperada):
    n_values = np.array([result[0] for result in resultados])
    tiempos = np.array([result[1] for result in resultados])
    # Ajustar los datos a una función de complejidad polinómica
    coeficientes = np.polyfit(n_values, tiempos, deg=len(complejidad_esperada) - 1)
    # Imprimir los coeficientes obtenidos
    print(f"Coeficientes del ajuste: {coeficientes}")
    # Comparar con la complejidad esperada
    print(f"Complejidad esperada: {complejidad_esperada}")
    # Ejemplo de cómo calcular la complejidad ajustada
    complejidad_ajustada = f"{coeficientes[-1]:.2e}"
    for i in range(len(coeficientes) - 1):
        complejidad_ajustada += f" + {coeficientes[i]:.2e} * n^{len(coeficientes) - i - 2}"
    print(f"Complejidad ajustada: {complejidad_ajustada}")
    return coeficientes

# Realizamos los cálculos para Ordenación por Inserción con inicialización ascendente
print("****Ordenación por Inserción con inicialización ascendente****")
resultados_ascendente_ins = calcular_complejidad_empirica(ins_sort, 1, 10, 1000, 10)
ajustar_complejidad(resultados_ascendente_ins, "O(n^2)")
print("\n\n")
# Realizamos los cálculos para Ordenación Shell con inicialización ascendente
print("****Ordenación Shell con inicialización ascendente****")
resultados_ascendente_shell = calcular_complejidad_empirica(shell_sort_hibbard, 1, 10, 1000, 10)
ajustar_complejidad(resultados_ascendente_shell, "O(?)")
print("\n\n")
# Realizamos los cálculos para Ordenación por Inserción con inicialización descendente
print("****Ordenación por Inserción con inicialización ascendente****")
resultados_descendente_ins = calcular_complejidad_empirica(ins_sort, 1, 10, 1000, 10)
ajustar_complejidad(resultados_descendente_ins, "O(n^2)")
print("\n\n")
# Realizamos los cálculos para Ordenación Shell con inicialización descendente
print("****Ordenación Shell con inicialización ascendente****")
resultados_descendente_shell = calcular_complejidad_empirica(shell_sort_hibbard, 1, 10, 1000, 10)
ajustar_complejidad(resultados_descendente_shell, "O(?)")
print("\n\n")
# Realizamos los cálculos para Ordenación por Inserción con inicialización aleatoria
print("****Ordenación por Inserción con inicialización ascendente****")
resultados_aleatoria_ins = calcular_complejidad_empirica(ins_sort, 1, 10, 1000, 10)
ajustar_complejidad(resultados_aleatoria_ins, "O(n^2)")
print("\n\n")
# Realizamos los cálculos para Ordenación Shell con inicialización aleatoria
print("****Ordenación Shell con inicialización ascendente****")
resultados_aleatoria_shell = calcular_complejidad_empirica(shell_sort_hibbard, 1, 10, 1000, 10)
ajustar_complejidad(resultados_aleatoria_shell, "O(?)")
print("\n\n")
'''