import numpy as np

from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import LSTM
import keras

# Fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    y = series[window_size:]
    X = [series[index:index + window_size] for index in range(len(y))]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y

# Build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


# Return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    for item in punctuation:
        text = text.replace(item, '')
    return text

# Fill out the function below that transforms the input text and window-size into a set of input/output
# pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    for i in range(int((len(text)) / step_size)):
        i_step = i * step_size
        if i_step + window_size < len(text):
            inputs.append(text[i_step:i_step + window_size])
            outputs.append(text[i_step + window_size])
    return inputs, outputs

# Build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation("softmax"))
    return model
