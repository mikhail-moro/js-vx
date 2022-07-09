import json
import random
import numpy as np
from tensorflow import keras

with open('data.json', 'r') as data:
    examples = json.load(data)

    lst_nums = []
    lst_ress = []

    lst = examples['train']
    random.shuffle(lst)

    for i in examples['train']:
        lst_nums.append([a for a in i[0]])
        lst_ress.append(i[1])



model = keras.Sequential()
hidden_layer_1 = keras.layers.Dense(20, activation = 'relu', batch_input_shape = (80, 20, 20), name = 'hidden_1')
model.add(hidden_layer_1)
hidden_layer_2 = keras.layers.Dense(10, activation = 'relu', name = 'hidden_2')
model.add(hidden_layer_2)
output_layer = keras.layers.Dense(1, activation = 'sigmoid', name = 'end')
model.add(output_layer)
sgd = keras.optimizers.SGD(learning_rate = 0.01)
model.compile(optimizer = sgd, loss = 'mse')

model.fit(np.array(lst_nums[:80]), np.array(lst_ress[:80]), epochs = 50, steps_per_epoch = 20)
results = model.predict(np.array(lst_nums[80:]), steps = 1, verbose = 0)

for i in results:
    print(i)

for layer in model.layers:
    print(layer.get_config(), layer.get_weights())
