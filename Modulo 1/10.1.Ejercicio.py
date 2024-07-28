# Problema:
# Escriba un programa que calcule el valor de x para el que la función f(x) = x^2 - 6x + 3 obtiene su menor resultado.
# Centre la búsqueda en el rango [-9, 9] sólo con valores enteros.

# Instrucciones:
# 1. Defina la función f(x) = x^2 - 6x + 3.
# 2. Inicialice una variable para almacenar el valor mínimo de la función (min_f) y otra para el valor de x correspondiente (min_x).
# 3. Use un bucle for para iterar sobre los valores enteros de x en el rango [-9, 9].
# 4. Para cada valor de x, evalúe la función f(x).
# 5. Compare el resultado con min_f. Si es menor, actualice min_f y min_x.
# 6. Imprima el valor de x que minimiza f(x) y el valor mínimo de f(x).



# 1. Definición de la función f(x)
def f(x):
    return x**2 - 6*x + 3

# 2. Inicialización de variables
min_f = float('inf')  # Inicializamos con infinito para asegurar que el primer valor sea menor
min_x = None 

# 3. Bucle for para iterar sobre el rango
for x in range(-9, 10):  # range(-9, 10) incluye los valores desde -9 hasta 9

    # 4. Evaluación de la función
    valor_f = f(x)

    # 5. Comparación y actualización del mínimo
    if valor_f < min_f:
        min_f = valor_f
        min_x = x

# Imprimir el resultado
print(f'El valor de x que minimiza f(x) es: {min_x}')
print(f'El valor mínimo de f(x) es: {min_f}') 
