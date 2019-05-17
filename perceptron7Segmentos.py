import tflearn
from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression

#Tendremos 7 entradas (una por cada LED activado en el display)
#Significado de las posiciones de cada valor del array de entrada X
# 0  Segmento superior del display
#1 2 Segmentos laterales superiores del display
# 3  Segmento central del display
#4 5 Segmentos laterales inferiores del display
# 6  Segmento inferior del display
#Ejemplo: [0,0,1,0,0,1,0] -> display con los LEDs 2 y 5 encendidos, por lo cual está representando un 1

X = [
    [1,1,1,0,1,1,1], #0
    [0,0,1,0,0,1,0], #1
    [1,0,1,1,1,0,1], #2
    [1,0,1,1,0,1,1], #3
    [0,1,1,1,0,1,0], #4
    [1,1,0,1,0,1,1], #5
    [1,1,0,1,1,1,1], #6
    [1,0,1,0,0,1,0], #7
    [1,1,1,1,1,1,1], #8
    [1,1,1,1,0,1,1], #9
    ]

# Tendremos una salida diferente para cada número a reconocer (por tanto, tendremos 10 salidas)
# -la posición 0 activada de la salida indica que el display está mostrando un 0, 
# -la posición 9 activada de la salida indica que el display está mostrando un 9,

Y = [
    [1,0,0,0,0,0,0,0,0,0], #0
    [0,1,0,0,0,0,0,0,0,0], #1
    [0,0,1,0,0,0,0,0,0,0], #2
    [0,0,0,1,0,0,0,0,0,0], #3
    [0,0,0,0,1,0,0,0,0,0], #4
    [0,0,0,0,0,1,0,0,0,0], #5
    [0,0,0,0,0,0,1,0,0,0], #6
    [0,0,0,0,0,0,0,1,0,0], #7
    [0,0,0,0,0,0,0,0,1,0], #8
    [0,0,0,0,0,0,0,0,0,1], #9
    ]

tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

input_layer = input_data(shape=[None, 7])
#[None, 7] significa que se puede entrenar la red con cualquier número de ejemplos, tales que cada ejemplo tenga 7 características. 
#Si sólo quisiéramos entrenar la red con 4 ejemplos escribiríamos shape=[4, 7]
hidden_layer = fully_connected(input_layer , 7, activation='leaky_relu')#, activation='relu') 
hidden_layer = tflearn.dropout(hidden_layer, 1)
output_layer = fully_connected(hidden_layer, 10, activation='softmax')

#Definimos el tipo de regresor que realizará la "back propagation" y entrenará nuestro modelo 
#Nosotros usaremos el "ADAM" como método de optimización y la "Categorical cross entropy" como función de error (loss function)
#La velocidad de aprendizaje del algoritmo (learning_rate) es 0.08
regression = regression(output_layer , optimizer='adam', loss='categorical_crossentropy', learning_rate=0.08)
#Finalmente defininimos nuestra "deep neural network" en tflearn usando DNN().
model = DNN(regression)
#Con model.fit(), Tensorflow entrenará a la red 5000 veces con los datos suministrados.
model.fit(X, Y, n_epoch=5000, show_metric=True)


#Si durante el entrenamiento, en la salida obtenemos un valor bajo para el parámetro "loss" 
#y un valor alto para el parámetro "accuracy" (precisión), entonces la red habrá aprendido correctamente
#Comprobar si el perceptrón ha aprendido todos los casos

i=0
binaryOutput=[]
for x in enumerate(X):
    maxValue=max(model.predict(X)[i])
    for out in model.predict(X)[i]:
        binaryOutput.append(1 if out==maxValue else 0)
    print(model.predict(X)[i])
    print ('Salida aprendida: ', Y[i])
    print ('Salida recordada: ', binaryOutput)
    binaryOutput.clear()
    i+=1