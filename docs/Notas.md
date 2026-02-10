# Simulacion_Gillingham: Notas internas del Proyecto

## Pendientes inmediatos

- Conectar satisfactoriamente config.py con simulacion.py
- Iniciar estimacion.py
- Documentar mejor (tal vez al finalizar cosillas)
- Mejorar con librerias qué hacen los mismos procesos que escribo desde cero como 'quantecon'


## Pendientes generales o futuro del proyecto

La simulación parece converger de 3 a 12 minutos, dependiendo el nivel de desequilibrio en el que empieza el mercado. La guess inicial parece disiminuir con éxito el tiempo de convergencia.

El modelo que he estado tratando tiene 3 marcas y 15 años de edad, pero el mundo real tiene muchisimas marcas o tipos de coches. Esto llama a algoritmos más eficientes o mayor poder computacional.

En el marco de poder computacional, tengo pendiente incluir jax para mandar todas las funciones al GPU. En tanto a algoritmos, hay algunos procesos que pude resolver con algoritmos que usan el paper, principalmente el de Newton para encontrar los puntos fijos de EV de forma más rápida.


---
