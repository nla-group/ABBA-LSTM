import numpy as np
import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)
import keras as K
import tensorflow as tf
import copy

class VanillaLSTM_keras(object):
    """ Vanilla LSTM implementation using keras """

    def __init__(self, num_layers=2, cells_per_layer=50, dropout=0.5, seed=None, stateful=True, lag=5):
        """
        Initialise and build the model
        """
        self.num_layers = num_layers
        self.cells_per_layer = cells_per_layer
        self.dropout = dropout
        self.seed = seed

        self.stateful = stateful
        self.lag = lag

        if seed != None:
            #tf.compat.v1.set_random_seed(seed)
            #session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            #sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
            #k.set_session(sess)
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

        # Reshape
        self.sequence = self.sequence.reshape(1, -1, self.features)

        self.model = build_Keras_LSTM(self.num_layers, self.cells_per_layer, self.lag, self.features, self.stateful, self.seed, self.dropout)

        if self.features != 1:
            self.model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.Adam())
        else:
            self.model.compile(loss='mse', optimizer=K.optimizers.Adam())

    def construct_training_index(self, debug=False):
        """
        Construct training index (compatible with model) from sequence of vectors of dimension d,
        """
        n = self.sequence.shape[1]
        self.index = []
        if self.stateful:
            # Create groups
            self.num_augs = min(self.lag, n - self.lag)
            for el in range(self.num_augs):
                self.index.append(np.arange(el, n - self.lag, self.lag))
        else:
            self.num_augs = 1
            self.index = np.arange(0, n - self.lag, 1)

    def train(self, patience=100, max_epoch=100000, acceptable_loss=np.inf, weight_restarts=False, debug=False):
        """
        Train the model on the constructed training data
        """
        ########################################################################
        # Weight restarts
        ########################################################################
        if weight_restarts:
            weight_restarts = 10
            store_weights = [0]*weight_restarts
            initial_loss = [0]*weight_restarts
            for i in range(weight_restarts):
                if self.stateful:
                    h = self.model.fit(self.sequence[:, 0:self.lag, :], self.sequence[:, self.lag, :], epochs=1, batch_size=1, verbose=0, shuffle=False)
                    initial_loss[i] = (h.history['loss'])[-1]
                    self.model.reset_states()
                    store_weights[i] = self.model.get_weights()

                    # quick hack to reinitialise weights
                    json_string = self.model.to_json()
                    self.model = model_from_json(json_string)
                    if self.features != 1:
                        self.model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.Adam())
                    else:
                        self.model.compile(loss='mse', optimizer=K.optimizers.Adam())
                else:
                    h = self.model.fit(self.sequence[:, 0:self.lag, :], self.sequence[:, self.lag, :], epochs=1, batch_size=1, verbose=0, shuffle=False) # no shuffling to remove randomness
                    initial_loss[i] = (h.history['loss'])[-1]
                    store_weights[i] = self.model.get_weights()
                    self.model.reset_states()

                    # quick hack to reinitialise weights
                    json_string = self.model.to_json()
                    self.model = K.models.model_from_json(json_string)
                    if isinstance(self.abba, ABBA):
                        self.model.compile(loss='categorical_crossentropy', optimizer=Adam())
                    else:
                        self.model.compile(loss='mse', optimizer=Adam())
            if debug:
                print('Initial loss:', initial_loss)
            m = np.argmin(initial_loss)
            self.model.set_weights(store_weights[int(m)])
            del store_weights


        ########################################################################
        # Train
        ########################################################################
        vec_loss = np.zeros(max_epoch)
        min_loss = np.inf
        min_loss_ind = np.inf
        losses = [0]*self.num_augs
        if self.stateful: # no shuffle and reset state manually
            for iter in range(max_epoch):
                rint = np.random.permutation(self.num_augs)
                for r in rint:
                    loss_sum = 0
                    for i in self.index[r]:
                        h = self.model.fit(self.sequence[:, i:i+self.lag, :], self.sequence[:, i+self.lag, :], epochs=1, batch_size=1, verbose=0, shuffle=False)
                        loss_sum += ((h.history['loss'])[-1])**2
                    losses[r] = loss_sum/len(self.index[r])
                    self.model.reset_states()
                vec_loss[iter] = np.mean(losses)

                if vec_loss[iter] >= min_loss:
                    if iter - min_loss_ind >= patience and min_loss < acceptable_loss:
                        break
                else:
                    min_loss = vec_loss[iter]
                    old_weights = self.model.get_weights()
                    min_loss_ind = iter

        else: # shuffle in fit
            for iter in range(max_epoch):
                loss_sum = 0
                for i in np.random.permutation(len(self.index)):
                    h = self.model.fit(self.sequence[:, i:i+self.lag, :], self.sequence[:, i+self.lag, :], epochs=1, batch_size=1, verbose=0, shuffle=True)
                    self.model.reset_states()
                    loss_sum += ((h.history['loss'])[-1])**2

                vec_loss[iter] = loss_sum/len(self.index)

                if vec_loss[iter] >= min_loss:
                    if iter - min_loss_ind >= patience and min_loss < acceptable_loss:
                        break
                else:
                    min_loss = (h.history['loss'])[-1]
                    old_weights = self.model.get_weights()
                    min_loss_ind = iter

        self.model.reset_states()
        self.model.set_weights(old_weights)
        self.epoch = iter+1
        self.loss = vec_loss[0:iter+1]


    def forecast(self, k, randomize=False, debug=False):
        """
        Make k step forecast into the future.
        """
        prediction = copy.deepcopy(self.sequence)
        # Recursively make k one-step forecasts
        for ind in range(self.sequence.shape[1], self.sequence.shape[1] + k):
            # Build data to feed into model
            if self.stateful:
                index = np.arange(ind%self.lag, ind, self.lag)
            else:
                index = [ind - self.lag]

            # Feed through model
            for i in index:
                p = self.model.predict(prediction[:, i:i+self.lag, :], batch_size = 1)

            # Convert output
            if self.features != 1:
                if randomize:
                    idx = np.random.choice(range(self.features), p=(p.ravel()))
                else:
                    idx = np.argmax(p.ravel())
                # Add forecast result to appropriate vectors.
                pred = np.zeros([1, 1, self.features])
                pred[0, 0, idx] = 1
            else:
                pred = np.array(float(p)).reshape([1, -1, 1])

            prediction = np.concatenate([prediction, pred], axis=1)
            # reset states in case stateless
            self.model.reset_states()

        if self.features != 1:
            return prediction.reshape(-1, self.features)
        else:
            return prediction.reshape(-1)

################################################################################
################################################################################
################################################################################

def build_Keras_LSTM(num_layers, cells_per_layer, lag, features, stateful, seed, dropout):
    model = K.models.Sequential()
    for index in range(num_layers):
        if index == 0:
            if num_layers == 1:
                if seed:
                    model.add(K.layers.LSTM(cells_per_layer, batch_input_shape=(1, lag, features), recurrent_activation='tanh', stateful=stateful, return_sequences=False, kernel_initializer=K.initializers.glorot_uniform(seed=seed), recurrent_initializer=K.initializers.Orthogonal(seed=seed)))
                    model.add(K.layers.Dropout(dropout, seed=seed))
                else:
                    model.add(K.layers.LSTM(cells_per_layer, batch_input_shape=(1, lag, features), recurrent_activation='tanh', stateful=stateful, return_sequences=False))
                    model.add(K.layers.Dropout(dropout))
            else:
                if seed:
                    model.add(K.layers.LSTM(cells_per_layer, batch_input_shape=(1, lag, features), recurrent_activation='tanh', stateful=stateful, return_sequences=True, kernel_initializer=K.initializers.glorot_uniform(seed=seed), recurrent_initializer=K.initializers.Orthogonal(seed=seed)))
                    model.add(K.layers.Dropout(dropout, seed=seed))
                else:
                    model.add(K.layers.LSTM(cells_per_layer, batch_input_shape=(1, lag, features), recurrent_activation='tanh', stateful=stateful, return_sequences=True))
                    model.add(K.layers.Dropout(dropout))
        elif index == num_layers-1:
            if seed:
                model.add(K.layers.LSTM(cells_per_layer, stateful=stateful, recurrent_activation='tanh', return_sequences=False, kernel_initializer=K.initializers.glorot_uniform(seed=seed), recurrent_initializer=K.initializers.Orthogonal(seed=seed)))
                model.add(K.layers.Dropout(dropout, seed=seed))
            else:
                model.add(K.layers.LSTM(cells_per_layer, stateful=stateful, recurrent_activation='tanh', return_sequences=False))
                model.add(K.layers.Dropout(dropout))
        else:
            if seed:
                model.add(K.layers.LSTM(cells_per_layer, stateful=stateful, recurrent_activation='tanh', return_sequences=True, kernel_initializer=K.initializers.glorot_uniform(seed=seed), recurrent_initializer=K.initializers.Orthogonal(seed=seed)))
                model.add(K.layers.Dropout(dropout, seed=seed))
            else:
                model.add(K.layers.LSTM(cells_per_layer, stateful=stateful, recurrent_activation='tanh', return_sequences=True, dropout=dropout, recurrent_dropout=dropout))
                model.add(K.layers.Dropout(dropout))

    if seed:
        model.add(K.layers.Dense(features, kernel_initializer=glorot_uniform(seed=0)))
    else:
        model.add(K.layers.Dense(features))

    if features != 1:
        model.add(K.layers.Activation('softmax'))

    return model
