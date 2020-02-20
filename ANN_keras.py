import numpy as np
import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)
import keras as K
import copy

class ANN_keras(object):
    """ ANN implementation using keras """

    def __init__(self, num_layers=2, neurons_per_layer=50, dropout=0.5, lag=5):
        """
        Initialise and build the model
        """
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.dropout = dropout
        self.lag = lag


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

        self.model = K.models.Sequential()
        for index in range(self.num_layers):
            if index == 0:
                self.model.add(K.layers.Dense(self.neurons_per_layer, input_shape=(self.lag, self.features), activation='relu'))
                self.model.add(K.layers.Dropout(self.dropout))
            else:
                self.model.add(K.layers.Dense(self.neurons_per_layer, activation='relu'))
                self.model.add(K.layers.Dropout(self.dropout))

        self.model.add(K.layers.Dense(self.features))

        if self.features != 1:
            self.model.add(K.layers.Activation('softmax'))
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
