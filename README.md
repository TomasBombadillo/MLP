# Multi-Layer Perceptron(MLP) - Entrenando XOR
Esta es la implementación del perceptrón multicapa usando la teoría. 
MPLlib.py es el programa que crea la clase Perceptron y todas sus características.
exec.py es el programaque llama a MPLlib para crear la neurona y llama los métodos necesariosapra analizar los resultados y entrenar la red.

### Prerequisitos
Se necesita la librería Numpy, de esta librería sale la mayor parte de operaciones con matrices y operaciones de elementos. Además se necesita la librería matplotlib.pyplot, para poder graficar error vs tiempo.

Para usarlo sólo es necesario entrar en la carpeta MPL, asegurarse que están los dos programas MPLlib.py y exec.py.
para correrlo, se debe ejecutar el siguiente comando:
```
python exec.py
```
Al instante empezaráa mostrar los resultados del entrenamiento, es decir la época, la rata de aprendizaje y el porcentaje de error.
Al acabar, ya sea porque corrió el número máximo de épocas o porque el error de entrenamiento llegó por debajo de $10^{-10}$, se mostrará una gráfica que ilustra el cambio del error mientras corrían las épocas, al cerrar la gráfica, se mostrará el resultado final, es decir el vector de salida, que idealmente debe ser el siguiente:
$$\begin{bmatrix} 0  \\ 1 \\ 1 \\0 \end{bmatrix}$$
pero muy pocas veces da esto exactamente.
### Nota
Puede que el programa logre entrenar el XOR exitosamente a la primera ejecución, pero puede darse el caso que no lo haga en las primeras iteraciones, en este caso es necesario ejecutarlo hasta que funcione, debería completar un entrenamiento exitoso en los primeros 5 intentos.
