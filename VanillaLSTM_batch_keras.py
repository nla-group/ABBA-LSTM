import numpy as np
import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)
import keras as K
import copy

class VanillaLSTM_batch_keras(object):
    """ Vanilla LSTM implementation using keras """

    def __init__(self, num_layers=2, cells_per_layer=50, dropout=0.5, seed=None, lag=5):
        """
        Initialise and build the model
        """
        self.num_layers = num_layers
        self.cells_per_layer = cells_per_layer
        self.dropout = dropout
        self.seed = seed
        self.lag = lag

        if seed != None:
            np.random.seed(seed)


    def build(self, sequence, debug=False):
        """
        Build model
        """
        self.sequence = sequence

        # Sequence either list of lists or a list.
        if sequence.ndim != 1:
            self.features = len(sequence[0])
        else:
            self.features = 1

        self.model = build_Keras_LSTM(self.num_layers, self.cells_per_layer, self.lag, self.features,  self.dropout)

        if self.features != 1:
            self.model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.Adam())
        else:
            self.model.compile(loss='mse', optimizer=K.optimizers.Adam())

    def construct_training_index(self, debug=False):
        """
        Construct training index (compatible with model) from sequence of vectors of dimension d,
        """
        self.n = self.sequence.shape[0]
        x = []
        y = []

        for i in range(self.n - self.lag):
            x.append(self.sequence[i:i+self.lag])
            y.append(self.sequence[i+self.lag])

        # batch, lag, dimension
        self.x_train = np.array(x).reshape(-1, self.lag, self.features)
        self.y_train = np.array(y).reshape(-1, self.features)

    def train(self, patience=100, max_epoch=100000, acceptable_loss=np.inf, batch_size=128, debug=False):
        """
        Train the model on the constructed training data
        """
        es = K.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
        history = self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=max_epoch, callbacks=[es], verbose=0)


    def forecast(self, k, randomize=False, debug=False):
        """
        Make k step forecast into the future.
        """
        prediction = copy.deepcopy(self.sequence)
        # Recursively make k one-step forecasts
        for ind in range(self.n, self.n + k):
            inp = prediction[-self.lag:].reshape(-1, self.lag, self.features)
            pred = self.model.predict(inp)[0]
            if self.features != 1:
                idx = np.argmax(pred)
                pred = np.zeros([1, self.features])
                pred[0, idx] = 1
                prediction = np.concatenate([prediction, pred])
            else:
                prediction = np.hstack([prediction, pred])

        return prediction

################################################################################
################################################################################
################################################################################

def build_Keras_LSTM(num_layers, cells_per_layer, lag, features,dropout):
    model = K.models.Sequential()
    for index in range(num_layers):
        if index == 0:
            if num_layers == 1:
                model.add(K.layers.LSTM(cells_per_layer, input_shape=(lag, features), recurrent_activation='tanh', return_sequences=False))
                model.add(K.layers.Dropout(dropout))
            else:
                model.add(K.layers.LSTM(cells_per_layer, input_shape=(lag, features), recurrent_activation='tanh', return_sequences=True))
                model.add(K.layers.Dropout(dropout))
        elif index == num_layers-1:
            model.add(K.layers.LSTM(cells_per_layer, recurrent_activation='tanh', return_sequences=False))
            model.add(K.layers.Dropout(dropout))
        else:
            model.add(K.layers.LSTM(cells_per_layer, recurrent_activation='tanh', return_sequences=True))
            model.add(K.layers.Dropout(dropout))

    model.add(K.layers.Dense(features))

    if features != 1:
        model.add(K.layers.Activation('softmax'))
    return model
