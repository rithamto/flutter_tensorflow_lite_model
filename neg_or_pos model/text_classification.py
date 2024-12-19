import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers # type: ignore
from tensorflow.keras import losses # type: ignore


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# dataset = tf.keras.utils.get_file(
#     "aclImdb_v1", url, untar=True, cache_dir=".", cache_subdir=""
# )

dataset_dir = "./aclImdb_v1/aclImdb"

remove_dir = os.path.join(dataset_dir + "/train", 'unsup')
shutil.rmtree(remove_dir)

batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    dataset_dir + "/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=seed,
)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    dataset_dir + "/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=seed,
)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    dataset_dir + "/test", batch_size=batch_size
)

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16

model = tf.keras.Sequential(
    [
        layers.Embedding(max_features, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.summary()

model.compile(
    loss=losses.BinaryCrossentropy(),
    optimizer="adam",
    metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)],
)

epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

loss, accuracy = model.evaluate(test_ds)
history_dict = history.history
history_dict.keys()

acc = history_dict["binary_accuracy"]
val_acc = history_dict["val_binary_accuracy"]
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]

epochs = range(1, len(acc) + 1)

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
metrics = export_model.evaluate(raw_test_ds, return_dict=True)
print(metrics)

examples = tf.constant([
  "bad",
  "good",
  "normal"
])

print(export_model.predict(examples))

converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
tflite_model = converter.convert()

# Lưu model vào file .tflite
with open('text_classification_model.tflite', 'wb') as f:
    f.write(tflite_model)