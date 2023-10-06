# Iván Moure Pérez, Pablo Hernández Martínez
# i.moure@udc.es, pablo.hernandez.martinez@udc.es

import random, time, numpy
from prettytable import PrettyTable

def hibbard_increments(array_length):
    increments = []
    k = 1
    gap = 2**k - 1
    while gap < array_length:
        increments.insert(0, gap)
        k += 1
        gap = 2**k - 1
    return increments

def shell_sort_hibbard(v):
    increments = hibbard_increments(len(v))
    return shell_sort_aux(v, increments)

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

def aleatorio(n):
    v = list(range(n))
    for i in v:
        v[i] = random.randint(-n, n)
    return v

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

def generar_vector(tipo, n):
    '''
    Vector tipo 1: ascendiente
    Vector tipo 2: descendiente
    Vector tipo 3: aleatorio
    '''
    if tipo in (1, 2):
        vector = numpy.arange(0, n)
        if tipo == 2:
            vector = numpy.flipud(vector)
    elif tipo == 3:
        vector = aleatorio(n)

    return vector

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

# Ejercicio 2
#test()

# Ejercicio 3

header_n  = ['n', 't(n) (ns)', 't(n)/n**0.8', ' t(n)/n', 't(n)/n**1.2']
header_n2 = ['n', 't(n) (ns)', 't(n)/n**1.8', ' t(n)/n**2.0', 't(n)/n**2.2']

ascIns,  ascShell  = PrettyTable(), PrettyTable()
descIns, descShell = PrettyTable(), PrettyTable()
randIns, randShell = PrettyTable(), PrettyTable()

ascIns.field_names,  ascShell.field_names  = header_n, header_n
descIns.field_names, descShell.field_names = header_n2, header_n
randIns.field_names, randShell.field_names = header_n2, header_n

n = 128
for i in range(8):
    vector = generar_vector(1, n)

    t = calcular_tiempo(ins_sort, vector, 1)
    ascIns.add_row([n, (str(t) + '*' if type(t) == float else t), t/n**0.8, t/n, t/n**1.2])

    t = calcular_tiempo(shell_sort_hibbard, vector, 1)
    ascShell.add_row([n, (str(t) + '*' if type(t) == float else t), t/n**0.8, t/n, t/n**1.2])

    n *= 2

n = 128
for i in range(8):
    vector = generar_vector(2, n)

    t = calcular_tiempo(ins_sort, vector, 2)
    descIns.add_row([n, (str(t) + '*' if type(t) == float else t), t/n**1.8, t/n**2.0, t/n**2.2])

    t = calcular_tiempo(shell_sort_hibbard, vector, 2)
    descShell.add_row([n, (str(t) + '*' if type(t) == float else t), t/n**0.8, t/n, t/n**1.2])

    n *= 2

n = 128 
for i in range(8):
    vector = generar_vector(3, n)

    t = calcular_tiempo(ins_sort, vector, 3)
    randIns.add_row([n, (str(t) + '*' if type(t) == float else t), t/n**1.8, t/n**2.0, t/n**2.2])

    t = calcular_tiempo(shell_sort_hibbard, vector, 3)
    randShell.add_row([n, (str(t) + '*' if type(t) == float else t), t/n**0.8, t/n, t/n**1.2])

    n *= 2

print()
print('Ordenacion por inserción con inicialización ascendiente.')
print(ascIns)
print()

print('Ordenacion Shell con inicialización ascendiente.')
print(ascShell)
print()

print('Ordenacion por inserción con inicialización descendiente.')
print(descIns)
print()

print('Ordenacion Shell con inicialización descendiente.')
print(descShell)
print()

print('Ordenacion por inserción con inicialización aleatoria.')
print(randIns)
print()

print('Ordenacion Shell con inicialización aleatoria.')
print(randShell)
print()

print('Los datos con un asterisco (*) indican que los tiempos fueron medidos de nuevo',
      'ya que no cumplían con el umbral de confianza de 500 microsegundos. El bucle que',
      'calcula la media se iteró 10 veces.', sep = '\n')
print()

# TODO: hay que modificar calcular_tiempo() porque cuando vuelve a medir el tiempo en el bucle
#       siempre usa un vector aleatorio en vez del apropiado. Para ello se ha creado la función
#       generar_vector()

# TODO: hay que encontrar las cotas ajustadas, que de eso va la práctica