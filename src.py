import numpy as np
import tensorflow as tf #installed tensorflow-hub and datasets also.
import tensorflow_hub as hub
import tensorflow_datasets as tfds
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)
train_example_batch, train_labels_batch = next(iter(train_data.batch(10))) #training
#embedding 
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer=hub.KerasLayer(embedding, input_shape = [], dtype=tf.string, trainable=True)
hub_layer(train_example_batch[:3])

#creating model
model = tf.Keras.Sequential()
model.add(hub_layer)

#output density layer
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

#compile model
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

#history of model
history = model.fit(train_data.shuffle(10000).batch(100), epochs = 25, validation_data = validation_data.batch(100), verbose=1)

results = model.evaluate(test_data.batch(100), verbose=2)
for name, values in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, values))