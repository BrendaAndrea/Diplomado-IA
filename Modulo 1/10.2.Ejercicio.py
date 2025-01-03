# Problema:
# Escriba un programa que procese una lista de estudiantes con sus respectivas calificaciones en diferentes materias,
# calcule el promedio de cada estudiante y luego ordene a los estudiantes por su promedio en orden descendente.

# Instrucciones:
# 1. Defina una lista de diccionarios, donde cada diccionario representa un estudiante y contiene su nombre y sus calificaciones en diferentes materias.
# 2. Calcule el promedio de calificaciones de cada estudiante.
# 3. Añada el promedio al diccionario de cada estudiante.
# 4. Ordene la lista de estudiantes por el promedio en orden descendente.
# 5. Imprima la lista ordenada de estudiantes con sus promedios.

# Definir la lista de diccionarios con los estudiantes y sus calificaciones
estudiantes = [
    {"nombre": "Alice", "matematicas": 85, "literatura": 78, "ciencia": 92},
    {"nombre": "Bob", "matematicas": 89, "literatura": 94, "ciencia": 85},
    {"nombre": "Charlie", "matematicas": 72, "literatura": 67, "ciencia": 80},
    {"nombre": "David", "matematicas": 90, "literatura": 88, "ciencia": 91},
    {"nombre": "Eva", "matematicas": 88, "literatura": 76, "ciencia": 85}
]

# Calcular el promedio de cada estudiante y ordenarlos
for estudiante in estudiantes:
    calificaciones = list(estudiante.values())[1:]  # Obtener calificaciones
    promedio = sum(calificaciones) / len(calificaciones)  
    estudiante["promedio"] = promedio  # Agregar el promedio al diccionario

estudiantes_ordenados = sorted(estudiantes, key=lambda x: x["promedio"], reverse=True)

# Imprimir resultados
print("Estudiantes ordenados por promedio (descendente):")
for estudiante in estudiantes_ordenados:
    print(f'{estudiante["nombre"]}: {estudiante["promedio"]:.2f}')  # Poner a 2 decimales
