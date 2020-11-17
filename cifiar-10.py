import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
tf.enable_eager_execution()

np.random.seed = 55
# dataset
dataset = cifar10
dataset_name = "cifar10"
(x_train, y_train), (x_test, y_test) = dataset.load_data()
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
batch_size = 512

if len(x_train.shape) == 4:
    img_channels = x_train.shape[3]
else:
    img_channels = 1

input_shape = (img_rows, img_cols, img_channels)
num_classes = len(np.unique(y_train))

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels).astype('float32') / 255.
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels).astype('float32') / 255.

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# model
model = load_model("test_keras.h5")

import time
n =1000
start = time.time()
pred = model.predict(x_test[:n])
print("time :", (time.time()-start)/n)
#print(pred)

loss, acc = model.evaluate(x_test, y_test)
print("normal model acc :", acc)

converter = tf.lite.TFLiteConverter.from_keras_model_file("test_keras.h5")

tflite_model = converter.convert()
with open("test_keras.tflite", "wb") as f:  # normal tflite model
    f.write(tflite_model)


converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset_gen():  # 이게 문제
    data = tf.data.Dataset.from_tensor_slices(x_train).batch(1)
    for input_value in data.take(batch_size):
    # Get sample input data as a numpy array in a method of your choosing.
        yield [input_value]

converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.uint8

tflite_model_quantized = converter.convert()

with open("test_keras_quantized.tflite", "wb") as f:  # save
    f.write(tflite_model_quantized)



interpreter = tf.lite.Interpreter("test_keras_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]

score = 0
#x_test = x_test.astype(np.uint8)

all_time = 0

for i in range(100):
    print(i)
    input_data = x_test[i:i + 1]  # shape과 data간 차원수 잘 맞추기
    #input_data = x_test[i:i + 1]
    #print(input_details[0])
    interpreter.set_tensor(input_details[0]["index"], input_data)
    start = time.time()
    interpreter.invoke()
    one_time =(time.time() - start) / n
    print("one_time",one_time )
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(i, " predict :", output_data)
    print(i, "answer :", np.argmax(y_test[i]))

    all_time = all_time + one_time
    if np.argmax(output_data) == np.argmax(y_test[i]):
        score += 1

print(score)
print(" predict :", np.argmax(output_data))
print("answer :", np.argmax(y_test[1]))
print("acc :", score/100)
print("all  time :",all_time)