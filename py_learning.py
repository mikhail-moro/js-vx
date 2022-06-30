import numpy as np
import randomuserlist
from tensorflow import keras


model = keras.Sequential()
hidden_layer = keras.layers.Dense(100, activation = 'relu', input_shape = (20, 20), name = 'hidden')
model.add(hidden_layer)
output_layer = keras.layers.Dense(2, activation = 'sigmoid', name = 'end')
model.add(output_layer)
sgd = keras.optimizers.SGD(learning_rate = 0.001)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy')
lst = randomuserlist.getData()

model.fit(np.array(lst[0]), np.array(lst[1]), epochs = 200, steps_per_epoch = 200)
results = model.predict(np.array(randomuserlist.getExample()), steps = 1, verbose = 0)


for i in results:
    print(i)

for layer in model.layers:
    print(layer.get_config(), layer.get_weights())