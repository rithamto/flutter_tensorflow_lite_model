import tensorflow as tf

# Tải mô hình từ file .h5
model = tf.keras.models.load_model('./model_crnn_word.h5')

# Chuyển đổi mô hình sang TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS 
]
converter._experimental_lower_tensor_list_ops = False
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()

# Lưu mô hình TFLite
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

