import numpy as np
import keras as k
import tensorflow as tf

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
            tf.compat.v1.set_random_seed(seed)
            # Force TensorFlow to use single thread.
            # Multiple threads are a potential source of non-reproducible results.
            # For further details, see: https://stackoverflow.com/questions/42022950/

            session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                          inter_op_parallelism_threads=1)
            sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
            k.set_session(sess)

            # prevent warning error about tensorflow build
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            np.random.seed(seed)


    def build(self, sequence, debug=False):
        """
        Build model
        """

        self.sequence = sequence

        # Sequence either list of lists or a list.
        if isinstance(sequence[0], type([])):
            self.features = len(sequence[0])
        else:
            self.features = 1

        self.model = build_Keras_LSTM(self.num_layers, self.cells_per_layer, self.lag, self.features, self.stateful, self.seed)

        if self.features != 1:
            self.model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.Adam())
        else:
            self.model.compile(loss='mse', optimizer=K.optimizers.Adam())

    def construct_training_data(self, debug=False):
        """
        Construct training data (compatible with model) from sequence of vectors of dimension d,
        """
        n = len(self.sequence)
        window = []
        if self.stateful:
            self.num_augs = min(self.lag, n - self.lag)
            for el in range(self.num_augs):
                w = []
                for i in np.arange(el, n - self.lag, self.lag):
                    w.append(self.sequence[i:i+self.lag+1])
                window.append(np.array(w).astype(float))
        else:
            self.num_augs = 1
            w = []
            for i in np.arange(0, n - self.lag, 1):
                w.append(self.sequence[i:i+self.lag+1])
            window.append(np.array(w).astype(float))

        # batch input of size (number of sequences, timesteps, data dimension)
        x = []
        for w in window:
            x.append(np.array(w[:, 0:-1]).reshape(-1, self.lag, self.features))

        y = []
        for w in window:
            # Unable to generalise y for both numeric and symbolic data
            if self.features != 1:
                y.append(np.array(w[:, -1, :]))
            else:
                y.append(np.array(w[:, -1]).reshape(-1, 1))

        self.x = x
        self.y = y

    def train(self, patience=100, max_epoch=100000, acceptable_loss=np.inf, debug=False):
        """
        Train the model on the constructed training data
        """
        ########################################################################
        # Weight restarts
        ########################################################################
        weight_restarts = 10
        store_weights = [0]*weight_restarts
        initial_loss = [0]*weight_restarts
        for i in range(weight_restarts):
            if self.stateful:
                h = self.model.fit(self.x[0], self.y[0], epochs=1, batch_size=1, verbose=0, shuffle=False)
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
                h = self.model.fit(self.x[0], self.y[0], epochs=1, batch_size=1, verbose=0, shuffle=False) # no shuffling to remove randomness
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
        loss = np.zeros(max_epoch)
        min_loss = np.inf
        min_loss_ind = np.inf
        losses = [0]*self.num_augs
        if self.stateful: # no shuffle and reset state manually
            for iter in range(epoch):

                rint = np.random.permutation(self.num_augs)

                for r in rint:
                    h = self.model.fit(self.x[r], self.y[r], epochs=1, batch_size=1, verbose=0, shuffle=False)
                    losses[r] = (h.history['loss'])[-1]
                    self.model.reset_states()
                loss[iter] = np.mean(losses)

                if loss[iter] >= min_loss:
                    if iter%100 == 0 and verbose:
                        print('iteration:', iter)
                    if iter - min_loss_ind >= self.patience and min_loss < self.acceptable_loss:
                        break
                else:
                    min_loss = loss[iter]
                    old_weights = self.model.get_weights()
                    if verbose:
                        print('iteration:', iter, 'loss:', min_loss)
                    min_loss_ind = iter
            if verbose:
                print('iteration:', iter)

        else: # shuffle in fit
            for iter in range(epoch):
                h = self.model.fit(x[0], y[0], epochs=1, batch_size=1, verbose=0, shuffle=True)
                loss[iter] = (h.history['loss'])[-1]

                if (h.history['loss'])[-1] >= min_loss:
                    if iter%100 == 0 and verbose:
                        print('iteration:', iter)
                    if iter - min_loss_ind >= self.patience and loss[iter] < self.acceptable_loss:
                        break
                else:
                    min_loss = (h.history['loss'])[-1]
                    old_weights = self.model.get_weights()
                    if verbose:
                        print('iteration:', iter, 'loss:', min_loss)
                    min_loss_ind = iter
            if verbose:
                print('iteration:', iter)

        if verbose:
            print('\nTraining complete! \n')
        self.model.reset_states()
        self.model.set_weights(old_weights)
        self.epoch = iter
        self.loss = loss[0:iter]


    def forecast(self, k, randomize=False, debug=False):
        """
        Make k step forecast into the future.
        """
        self.model.eval()

        if self.features == 1:
            prediction = self.sequence[::]
        else:
            prediction = self.sequence[::].tolist()

        # Recursively make fl one-step forecasts
        for ind in range(len(self.sequence), len(self.sequence) + k):

            # Build data to feed into model
            if self.stateful:
                window = []
                for i in np.arange(ind%self.lag, ind, self.lag):
                    window.append(prediction[i:i+self.lag])
            else:
                window = prediction[-self.lag:]
            pred_x =  np.array(window).astype(float)
            pred_x = np.array(pred_x).reshape(-1, self.lag, self.features)

            # Feed through model
            states = self.model.initialise_states()
            for el in pred_x:
                p, states = self.model.forward(torch.tensor(el).float(), (states[0].detach(), states[1].detach()))

            # Convert output
            if self.features == 1:
                softmax = torch.nn.Softmax(dim=-1)
                p = softmax(p).tolist()
                p = np.array(p)
                p /= p.sum()
                if randomize:
                    idx = np.random.choice(range(self.features), p=(p.ravel()))
                else:
                    idx = np.argmax(list(p), axis = 0)

                # Add forecast result to appropriate vectors.
                add = np.zeros([1, self.features])
                add[0, idx] = 1
                prediction.append((add.tolist())[0])
            else:
                prediction.append(float(p))

        return prediction

################################################################################
################################################################################
################################################################################

def build_Keras_LSTM(num_layers, cells_per_layer, lag, features, stateful, seed):
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
        model.add(Dense(self.features, kernel_initializer=glorot_uniform(seed=0)))
    else:
        model.add(Dense(self.features))

    if features != 1:
        model.add(Activation('softmax'))

    return model
