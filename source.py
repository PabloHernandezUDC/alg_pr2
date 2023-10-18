# Iván Moure Pérez, Pablo Hernández Martínez
# i.moure@udc.es, pablo.hernandez.martinez@udc.es

import random, time
import numpy as np
from math import log
from prettytable import PrettyTable

# Calculamos tiempo de ejecución (inicio)
totalStart = time.perf_counter_ns()

# EJERCICIO Nº1
print("\n\n***Ejercicio 1***\n")
print('En este ejercicio, se definen algoritmos de ordenación y funciones auxiliares que se utilizarán posteriormente para analizar su rendimiento.')

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
    # Creamos la tabla table_test
    table_test = PrettyTable()
    # Establecemos los nombres de las columnas de la tabla table_test
    table_test.field_names =['Tipo de inicialización','Ordenación','Resultado','Éxito o Fracaso']
    # El vector con el que trabajaremos será de longitud 'n'
    vector = aleatorio(n)
    result = ins_sort(vector.copy())
    # Añadimos una primera fila a la tabla table_test
    table_test.add_row(['Aleatoria','Inserción',result, 'Éxito' if result == sorted(vector) else 'Fracaso'])
    
    # Añadimos una segunda fila a la tabla table_test
    vector = sorted(vector, reverse = True)
    result = shell_sort_hibbard(vector.copy())
    table_test.add_row(['Descendiente','Shell',result,  'Éxito' if result == sorted(vector) else 'Fracaso'])
    
    #Mostramos la tabla resultante
    print(table_test)
   


# EJERCICIO Nº2
'''
En este ejercicio, se ejecuta la función test() para realizar pruebas de ordenación con vectores aleatorios.
'''
# Calculamos el tiempo de ejecución (inicio Ejercicio 2)
start_2 = time.perf_counter_ns()

print("\n\n***Ejercicio 2*** ")
test()

# Calculamos tiempo de ejecución (final Ejercicio 2)
finish_2 = time.perf_counter_ns()
# Mostramos el tiempo de ejecución
print(f'\nTiempo de ejecución del Ejercicio 2: {round((finish_2 - start_2) / (10**7), 2)} centésimas de segundo.')
print()


# EJERCICIO Nº3
'''
En este ejercicio, se realiza un análisis de rendimiento de los algoritmos de ordenación en diferentes configuraciones y se crean tablas
para registrar los resultados.
'''
# Calculamos el tiempo de ejecución (inicio Ejercicio 3)
start_3 = time.perf_counter_ns()

print("\n\n***Ejercicio 3*** ")
# Creamos las 6 tablas pedidas
ascIns,  ascShell  = PrettyTable(), PrettyTable()
descIns, descShell = PrettyTable(), PrettyTable()
randIns, randShell = PrettyTable(), PrettyTable()

# Establecemos los nombres de los distintos parámetros de las tablas
ascIns.title          = 'Ordenacion por inserción con inicialización ascendiente'
ascIns.field_names    = ['n', 't(n) (ns)', 't(n)/n', ' t(n)/n**1.04', 't(n)/n**1.08']
ascShell.title        = 'Ordenacion Shell con inicialización ascendiente'
ascShell.field_names  = ['n', 't(n) (ns)', 't(n)/n', ' t(n)/(n)*log(n/2)', 't(n)/n**1.2']

descIns.title         = 'Ordenacion por inserción con inicialización descendiente'
descIns.field_names   = ['n', 't(n) (ns)', 't/((n**1.8))', 't/((n**2))', 't/((n**2.2))']
descShell.title       = 'Ordenacion Shell con inicialización descendiente'
descShell.field_names = ['n', 't(n) (ns)', 't/((1/n)*(log(n)**2))', ' t/((n)*(log(n)**2))', 't/((n)*(log(2*n)))']

randIns.title         = 'Ordenacion por inserción con inicialización aleatoria'
randIns.field_names   = ['n', 't(n) (ns)', 't(n)/n**1.8', ' t(n)/n**2.0', 't(n)/n**2.2']
randShell.title       = 'Ordenacion Shell con inicialización aleatoria'
randShell.field_names = ['n', 't(n) (ns)', 't(n)/n', ' t(n)/n**1.09', 't(n)/n**1.18']


# A continuación, se realizan cálculos de tiempo y se llenan las tablas con los resultados...

n = 128
# Creamos un bucle para multiplicar n por 2 un total de 8 veces
for i in range(8):
    vector = generar_vector(1, n)
    # Ordenacion por inserción con inicialización ascendiente
    t = calcular_tiempo(ins_sort, vector, 1)
    ascIns.add_row([n, (str(t) + '*' if type(t) == float else t), t/n, t/n**1.04, t/n**1.08])
    # Ordenacion Shell con inicialización ascendiente
    t = calcular_tiempo(shell_sort_hibbard, vector, 1)
    ascShell.add_row([n, (str(t) + '*' if type(t) == float else t), t/n**1, t/((n)*(log(n/2))), t/n**1.2])

    n *= 2

n = 128
# Creamos un bucle para multiplicar n por 2 un total de 8 veces
for i in range(8):
    vector = generar_vector(2, n)
    # Ordenacion por inserción con inicialización descendiente
    t = calcular_tiempo(ins_sort, vector, 2)
    descIns.add_row([n,
                     (str(t) + '*' if type(t) == float else t),
                     t/((n**1.8)),
                     t/((n**2)),
                     t/((n**2.2))])

    # Ordenacion Shell con inicialización descendiente
    t = calcular_tiempo(shell_sort_hibbard, vector, 2)
    descShell.add_row([n,
                       (str(t) + '*' if type(t) == float else t),
                       t/((1/n)*(log(n)**2)),
                       t/((n)*(log(n)**2)),
                       t/((n)*(log(2*n)))])

    n *= 2

n = 128 
#  Creamos un bucle para multiplicar n por 2 un total de 8 veces
for i in range(8):
    vector = generar_vector(3, n)
    # Ordenacion por inserción con inicialización aleatoria
    t = calcular_tiempo(ins_sort, vector, 3)
    randIns.add_row([n, (str(t) + '*' if type(t) == float else t), t/n**1.8, t/n**2.0, t/n**2.2])
    # Ordenacion Shell con inicialización aleatoria
    t = calcular_tiempo(shell_sort_hibbard, vector, 3)
    randShell.add_row([n, (str(t) + '*' if type(t) == float else t), t/n, t/n**1.09, t/n**1.18])

    n *= 2

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
# Calculamos tiempo de ejecución (final Ejercicio 3)
finish_3 = time.perf_counter_ns()
# Mostramos el tiempo de ejecución
print(f'\nTiempo de ejecución del Ejercicio 3: {round((finish_3 - start_3) / (10**9), 2)}s.')
print()

# EJERCICIO Nº4
'''
En este ejercicio, se calcula empíricamente la complejidad de los algoritmos de ordenación y se ajustan los
resultados a la complejidad esperada.
'''
# Calculamos el tiempo de ejecución (inicio Ejercicio 4)
start_4 = time.perf_counter_ns()

print("\n\n***Ejercicio 4*** ")

# Con la función PrettyTable creamos la tabla en la que almacenaremos las complejidades
tabla_complejidades = PrettyTable()

# Escribimos el título de la tabla y el nombre de las columnas 
tabla_complejidades.title       = 'Tabla de complejidades'
tabla_complejidades.field_names = ['Nombre tabla' ,'Complejidad esperada', 'Complejidad empírica']

tabla_complejidades.add_row(['inserción-ascendiente' ,'O(n)'           ,'O(n^1.04)'])
tabla_complejidades.add_row(['Shell-ascendiente'     ,'O(n)'           ,'O(n*(log(n/2)))'])
tabla_complejidades.add_row(['inserción-descendiente','O((n^2)/2)'     ,'O((n^2)/2)'])
tabla_complejidades.add_row(['Shell-descendiente'    ,'O(n*(log^2(n)))','O(n*(log^2(n)))'])
tabla_complejidades.add_row(['inserción-aleatoria'   ,'O(n^2)'         ,'O(n^2)'])
tabla_complejidades.add_row(['Shell-aleatoria'       ,'O(n^2)'         ,'O(n^1.09)'])

print(tabla_complejidades)

# Calculamos tiempo de ejecución (final Ejercicio 4)
finish_4 = time.perf_counter_ns()
# Mostramos el tiempo de ejecución
print(f'\nTiempo de ejecución del Ejercicio 4: {round((finish_4 - start_4) / (10**7), 2)} centésimas de segundo.')
print()

# Calculamos tiempo de ejecución (final)
totalFinish = time.perf_counter_ns()
# Mostramos el tiempo de ejecución
print(f'\n\n\nTiempo total de ejecución del programa entero: {round((totalFinish - totalStart) / (10**9), 2)} segundos.')
print()