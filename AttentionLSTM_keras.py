import numpy as np
import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)
import keras as K
import copy

class AttentionLSTM_keras(object):
    """ Attention LSTM implementation using keras """

    def __init__(self, LSTM_cells=50, Dense_neurons = 50, lag=5):
        """
        Initialise and build the model
        """
        self.LSTM_cells = LSTM_cells
        self.Dense_neurons = Dense_neurons
        self.lag = lag

    def build(self, sequence, debug=False):
        """
        Build model
        """
        self.sequence = sequence
        # Sequence either list of lists or a list.
        if sequence.ndim != 1:
            self.features = len(sequence[0])
            warnings.warn('AttentionLSTM does not support ABBA representations')
        else:
            self.features = 1

        input_layer = K.layers.Input(shape=(self.lag, self.features), dtype='float32')
        x = K.layers.LSTM(self.LSTM_cells, return_sequences=True)(input_layer)
        attention_pre = K.layers.Dense(1)(x)
        attention_probs = K.layers.Softmax()(attention_pre)
        attention_mul = K.layers.Lambda(lambda x:x[0]*x[1])([attention_probs,x])
        x = K.layers.Flatten()(attention_mul)
        x = K.layers.Dense(self.Dense_neurons, activation='relu')(x)
        preds = K.layers.Dense(1, activation='linear')(x)
        self.model = K.models.Model(input_layer, preds)
        self.model.compile(loss='mse', optimizer=K.optimizers.Adam(),metrics=['mse'])


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
