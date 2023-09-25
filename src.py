import numpy as np
import tensorflow as tf #installed tensorflow-hub and datasets also.
import tensorflow_hub as hub
import tensorflow_datasets as tfds
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)
