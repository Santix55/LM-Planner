# Proyecto: Planificador basado en Modelos de Lenguaje

Este proyecto contiene **cuatro** archivos de Python, cada uno implementando una versión del programa planificador basado en modelos de lenguaje:

- **`basic-planner.py`**: Implementa la primera versión del programa.
- **`better-context.py`**: Añade un mejor prompt para el algoritmo y un historial de las acciones realizadas.
- **`window-planner.py`**: Introduce una ventana de contexto que permite al agente conocer no solo sus acciones previas, sino también todas sus respuestas previas.
- **`retry-answer.py`**: Implementa un mecanismo de autocorrección de errores en respuestas inválidas.

## Uso del Programa

Para ejecutar los programas, simplemente hay que correr el archivo correspondiente con un intérprete de Python, sin necesidad de proporcionar argumentos. No obstante, el programa requiere algunos parámetros ajustables que pueden editarse directamente desde el editor de código:

- **`path_domain`**: Ruta del archivo del dominio PDDL.
- **`path_problem`**: Ruta del archivo del problema PDDL.
- **`verbose`** (`True` o `False`): Define si se quiere mostrar la traza del algoritmo durante la ejecución.
- **`max_steps`**: Número máximo de pasos que puede realizar el planificador.

## Archivos de Ejemplo

Para probar el problema, en la carpeta **`./pddl/`** se pueden encontrar varios dominios y problemas de ejemplo para experimentar con los diferentes planificadores.



