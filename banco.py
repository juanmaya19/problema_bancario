import simpy
import numpy as np

# Parámetros del problema
HORAS_OPERACION = 8
MINUTOS_OPERACION = HORAS_OPERACION * 60
NUM_CAJEROS = 3
PROPORCION_RETIROS = 0.7
PROPORCION_PAGOS = 1 - PROPORCION_RETIROS
TIPOS_USUARIOS = {
    'retiro': {
        'rápido': {'tiempo_servicio': 1, 'tiempo_llegada': 1, 'probabilidad': 0.23},
        'normal': {'tiempo_servicio': 2, 'tiempo_llegada': 2, 'probabilidad': 0.40},
        'lento': {'tiempo_servicio': 3, 'tiempo_llegada': 3, 'probabilidad': 0.17},
        'muy_lento': {'tiempo_servicio': 4, 'tiempo_llegada': 3, 'probabilidad': 0.20},
    },
    'pago': {
        'rápido': {'tiempo_servicio': 3, 'tiempo_llegada': 1, 'probabilidad': 0.10},
        'normal': {'tiempo_servicio': 3, 'tiempo_llegada': 2, 'probabilidad': 0.20},
        'lento': {'tiempo_servicio': 5, 'tiempo_llegada': 3, 'probabilidad': 0.30},
        'muy_lento': {'tiempo_servicio': 7, 'tiempo_llegada': 4, 'probabilidad': 0.40},
    },
}

# Función para seleccionar el tipo de usuario basado en probabilidades
def seleccionar_tipo_usuario(accion):
    tipos = TIPOS_USUARIOS[accion]
    nombres_tipos = list(tipos.keys())
    probabilidades = [tipos[tipo]['probabilidad'] for tipo in nombres_tipos]
    return np.random.choice(nombres_tipos, p=probabilidades)

# Función para simular la llegada de usuarios
def llegada_usuarios(env, nombre_cajero, caja, accion, usuarios_atendidos, tiempos_servicio):
    while True:
        tipo_usuario = seleccionar_tipo_usuario(accion)
        tiempo_llegada = TIPOS_USUARIOS[accion][tipo_usuario]['tiempo_llegada']
        yield env.timeout(np.random.exponential(tiempo_llegada))
        
        env.process(atender_usuario(env, nombre_cajero, caja, tipo_usuario, accion, usuarios_atendidos, tiempos_servicio))

# Función para simular el servicio de los usuarios
def atender_usuario(env, nombre_cajero, caja, tipo_usuario, accion, usuarios_atendidos, tiempos_servicio):
    with caja.request() as request:
        yield request
        tiempo_servicio = TIPOS_USUARIOS[accion][tipo_usuario]['tiempo_servicio']
        yield env.timeout(np.random.exponential(tiempo_servicio))
        
        usuarios_atendidos[nombre_cajero][accion][tipo_usuario] += 1
        tiempos_servicio[nombre_cajero].append(env.now)

# Función principal para ejecutar la simulación
def ejecutar_simulacion():
    env = simpy.Environment()
    cajas = [simpy.Resource(env, capacity=1) for _ in range(NUM_CAJEROS)]
    
    usuarios_atendidos = {f'Cajero_{i+1}': {'retiro': {tipo: 0 for tipo in TIPOS_USUARIOS['retiro']},
                                            'pago': {tipo: 0 for tipo in TIPOS_USUARIOS['pago']}} for
                                            i in range(NUM_CAJEROS)}
    tiempos_servicio = {f'Cajero_{i+1}': [] for i in range(NUM_CAJEROS)}
    
    for i, caja in enumerate(cajas):
        nombre_cajero = f'Cajero_{i+1}'
        accion = 'retiro' if np.random.rand() < PROPORCION_RETIROS else 'pago'
        env.process(llegada_usuarios(env, nombre_cajero, caja, accion, usuarios_atendidos, tiempos_servicio))
    
    env.run(until=MINUTOS_OPERACION)
    
    return usuarios_atendidos, tiempos_servicio

# Ejecutar múltiples réplicas de la simulación
def replicar_simulacion(num_replicas=10):
    resultados = []
    for _ in range(num_replicas):
        resultados.append(ejecutar_simulacion())
    return resultados

# Analizar los resultados de las réplicas
def analizar_resultados(resultados):
    tiempos_promedio_cajero = {}
    usuarios_totales_tipo = {'retiro': {'rápido': [], 'normal': [], 'lento': [], 'muy_lento': []},
                             'pago': {'rápido': [], 'normal': [], 'lento': [], 'muy_lento': []}}

    for usuarios_atendidos, tiempos_servicio in resultados:
        for cajero, tiempos in tiempos_servicio.items():
            if tiempos:
                tiempos_promedio_cajero[cajero] = np.mean(tiempos)
            else:
                tiempos_promedio_cajero[cajero] = 0

        for cajero, usuarios in usuarios_atendidos.items():
            for accion, tipos in usuarios.items():
                for tipo, cantidad in tipos.items():
                    usuarios_totales_tipo[accion][tipo].append(cantidad)
    
    return tiempos_promedio_cajero, usuarios_totales_tipo

# Función para calcular el total de usuarios atendidos en una simulación
def total_usuarios(usuarios):
    total = 0
    for cajero, datos in usuarios.items():
        for accion, tipos in datos.items():
            for tipo, cantidad in tipos.items():
                total += cantidad
    return total



# Main
resultados = replicar_simulacion()
tiempos_promedio_cajero, usuarios_totales_tipo = analizar_resultados(resultados)

# Mostrar los resultados
print("1. Tiempos promedio de atención por cajero:")
for cajero, tiempo in tiempos_promedio_cajero.items():
    print(f"{cajero}: {tiempo:.2f} minutos")

cajero_menor_tiempo = min(tiempos_promedio_cajero, key=tiempos_promedio_cajero.get)
cajero_mayor_tiempo = max(tiempos_promedio_cajero, key=tiempos_promedio_cajero.get)
print(f"Cajero con menor tiempo promedio de atención: {cajero_menor_tiempo}")
print(f"Cajero con mayor tiempo promedio de atención: {cajero_mayor_tiempo}")

print("\n2. Promedio de usuarios atendidos por tipo en la totalidad de cajeros:")
for accion, tipos in usuarios_totales_tipo.items():
    for tipo, cantidades in tipos.items():
        print(f"{accion.capitalize()} {tipo}: Promedio = {np.mean(cantidades):.2f}, 
              Desviación estándar = {np.std(cantidades):.2f}")

print("\n3. Total de usuarios de cada tipo en cada una de las réplicas:")
for i, (usuarios_atendidos, _) in enumerate(resultados):
    print(f"Réplica {i+1}:")
    for accion, tipos in usuarios_atendidos.items():
        for tipo, cantidad in tipos.items():
            print(f"  {accion.capitalize()} {tipo}: {cantidad}")

# Encuentra el modelo con la menor cantidad total de usuarios atendidos
modelo_menor_usuarios = min(resultados, key=lambda x: total_usuarios(x[0]))

# Mostrar el modelo con menor cantidad de usuarios por tipo (en total)
print("\nModelo con menor cantidad de usuarios por tipo (en total):")
for cajero, datos in modelo_menor_usuarios[0].items():
    print(f"{cajero}:")
    for accion, tipos in datos.items():
        for tipo, cantidad in tipos.items():
            print(f"  {accion.capitalize()} {tipo}: {cantidad}")





print("\n4. Definir si es necesario crear un nuevo cajero:")
# Aquí se hace un análisis sobre los tiempos de espera y el promedio de usuarios atendidos
tiempo_maximo_aceptable = 5  # Suponiendo un tiempo máximo aceptable de 5 minutos
necesita_nuevo_cajero = any(tiempo > tiempo_maximo_aceptable for tiempo in tiempos_promedio_cajero.values())
if necesita_nuevo_cajero:
    print("Es necesario crear un nuevo cajero.")
else:
    print("No es necesario crear un nuevo cajero.")

print("\n5. Decidir cuántos cajeros deben ofrecer atención exclusiva para pagos y cuántos para retiros:")
usuarios_totales_retiros = sum(np.mean(usuarios_totales_tipo['retiro'][tipo]) for tipo in usuarios_totales_tipo['retiro'])
usuarios_totales_pagos = sum(np.mean(usuarios_totales_tipo['pago'][tipo]) for tipo in usuarios_totales_tipo['pago'])

proporcion_retiros = usuarios_totales_retiros / (usuarios_totales_retiros + usuarios_totales_pagos)
proporcion_pagos = usuarios_totales_pagos / (usuarios_totales_retiros + usuarios_totales_pagos)

cajeros_retiros = round(proporcion_retiros * NUM_CAJEROS)
cajeros_pagos = NUM_CAJEROS - cajeros_retiros

print(f"Se recomienda asignar {cajeros_retiros} cajeros exclusivos para retiros y {cajeros_pagos} cajeros exclusivos para pagos.")


