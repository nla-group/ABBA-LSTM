# Ignore future warnings caused by tensorflow and numpy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Check abba available to see if pip installed requirements.txt
import importlib
spec = importlib.util.find_spec("ABBA")
if spec is None:
    warning.warn("Try: pip install -r 'requirements.txt'")
from ABBA import ABBA as ABBA

# supress OpenMP warnings when specifying tensorflow threads
import os
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
os.environ['KMP_WARNINGS'] = 'off'
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.initializers import Orthogonal, glorot_uniform
from keras.optimizers import Adam
from keras import backend as k
from keras.models import model_from_json
import tensorflow as tf

# import all other modules
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
from matplotlib.pyplot import plot,title,xlabel,ylabel,legend,grid,style,xlim,ylim,axis,show
from collections import Counter

class LSTM_model(object):
    """
    LSTM_model class used to build and train networks of LSTM cells for time
    series prediction using numerical data and symbolic data via ABBA compression.

    Possible attributes
    -------------------
    num_layers                      - number of layers of stacked LSTMs
    cells_per_layer                 - number of LSTMs per layer
    dropout                         - amount of dropout after each LSTM layer
    ts                              - original time series
    abba                            - abba class for abba representation
    stateful                        - bool if keras 'stateful' LSTM is used
    l                               - lag parameter is the amount of recurrence
    mean                            - mean of ts
    std                             - standard deviation of ts
    normalised_data                 - z-normalised version of ts
    ABBA_representation_string      - ABBA string representation of ts
    centers                         - cluster centers from ABBA compression
    ABBA_representation_numeric     - ABBA numeric representation
    alphabet                        - alphabet used in ABBA compression
    training_data                   - data used for training network
    features                        - size of the alphabet
    model                           - network model
    patience                        - patience parameter for training
    acceptable_loss                 - acceptable loss for training
    optimizer                       - optimization algorithm used
    epoch                           - number of iterations used during training
    loss                            - list of loss value at each iteration
    start_prediction_ts             - numeric start prediction
    start_prediction_txt            - symbolic start prediction
    in_sample_ts                    - numeric in sample forecast
    in_sample_txt                   - symbolic in sample forecast
    out_of_sample_ts                - numeric out of sample forecast
    out_of_sample_txt               - symbolic out of sample forecast
    """

    def __init__(self, num_layers=2, cells_per_layer=50, dropout=0.5, seed=None):
        """
        Initialise class object. Read in shape of model.

        Parameters
        ----------
        num_layers - int
                Number of layers of stacked LSTMs

        cells_per_layer - int
                Number of LSTM cells per layer. Equally think of it as the
                dimension of the hidden layer.

        dropout - float
                Value in the interval [0, 1] controlling the amound of dropout
                after each layer of LSTMs.

        seed - int
                Seed for weight initialisation and shuffling order.
        """

        self.num_layers = num_layers
        self.cells_per_layer = cells_per_layer
        self.dropout = dropout
        self.seed = seed

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


    def build(self, ts, l=1, stateful=True, abba=None, verbose=True):
        """
        Build model, this function requires the time series and abba class object
        to understand input dimensions of the network.

        Parameter
        ---------
        ts - list
                List containing the timeseries

        l - int
                Timesteps back controls the length of the 'short term' memory. The
                number of recurrent steps when training the LSTM.

        stateful - bool
                True - keras 'stateful' LSTM, so the cell state information is passed
                between elements of the training set. The cell state must be manually
                reset and the order of the dataset is important.

        abba - ABBA class object
                If no abba object is given, then time series prediction uses
                numerical representation. Otherwise abba class is used for symbolic
                conversion and network is trainined on the symbolic data.

        verbose - bool
                True - print progress
                False - print nothing
                Note, does not supress printing from ABBA module, must supress when
                constructing ABBA class object.
        """

        # Read in parameters
        self.ts = ts
        self.abba = abba
        self.l = l
        self.stateful = stateful

        # Normalise time series
        self.mean = np.mean(ts)
        self.std = np.std(ts)
        normalised_data = (ts - self.mean)/self.std if self.std!=0 else ts - self.mean
        self.normalised_data = normalised_data

        # Construct ABBA representation
        if isinstance(self.abba, ABBA):
            if verbose:
                print('\nApplying ABBA compression! \n')
            # Apply abba transformation
            self.pieces = abba.compress(self.normalised_data)
            self.ABBA_representation_string, self.centers = abba.digitize(self.pieces)
            # Apply inverse transform.
            self.ABBA_representation_numerical = self.mean + np.dot(self.std, abba.inverse_transform(self.ABBA_representation_string, self.centers, self.normalised_data[0]))

            # One hot encode symbolic representation. Create list of all symbols
            # in case symbol does not occur in symbolic representation. (example:
            # flat line and insist k>1)
            self.alphabet = sorted([chr(97+i) for i in range(len(self.centers))])
            self.training_data = [[0 if char != letter else 1 for char in self.alphabet] for letter in self.ABBA_representation_string]

            # Calculate dimension of data
            self.features = len(self.alphabet)
            if verbose:
                print('\nABBA compression complete! \n')

        else:
            self.training_data = self.normalised_data
            self.features = 1

        # Build model.
        model = Sequential()
        for index in range(self.num_layers):
            if index == 0:
                if self.num_layers == 1:
                    if self.seed:
                        model.add(LSTM(self.cells_per_layer, batch_input_shape=(1, self.l, self.features), recurrent_activation='tanh', stateful=self.stateful, return_sequences=False, kernel_initializer=glorot_uniform(seed=self.seed), recurrent_initializer=Orthogonal(seed=self.seed)))
                        model.add(Dropout(self.dropout, seed=self.seed))
                    else:
                        model.add(LSTM(self.cells_per_layer, batch_input_shape=(1, self.l, self.features), recurrent_activation='tanh', stateful=self.stateful, return_sequences=False))
                        model.add(Dropout(self.dropout))
                else:
                    if self.seed:
                        model.add(LSTM(self.cells_per_layer, batch_input_shape=(1, self.l, self.features), recurrent_activation='tanh', stateful=self.stateful, return_sequences=True, kernel_initializer=glorot_uniform(seed=self.seed), recurrent_initializer=Orthogonal(seed=self.seed)))
                        model.add(Dropout(self.dropout, seed=self.seed))
                    else:
                        model.add(LSTM(self.cells_per_layer, batch_input_shape=(1, self.l, self.features), recurrent_activation='tanh', stateful=self.stateful, return_sequences=True))
                        model.add(Dropout(self.dropout))
            elif index == self.num_layers-1:
                if self.seed:
                    model.add(LSTM(self.cells_per_layer, stateful=self.stateful, recurrent_activation='tanh', return_sequences=False, kernel_initializer=glorot_uniform(seed=self.seed), recurrent_initializer=Orthogonal(seed=self.seed)))
                    model.add(Dropout(self.dropout, seed=self.seed))
                else:
                    model.add(LSTM(self.cells_per_layer, stateful=self.stateful, recurrent_activation='tanh', return_sequences=False))
                    model.add(Dropout(self.dropout))
            else:
                if self.seed:
                    model.add(LSTM(self.cells_per_layer, stateful=self.stateful, recurrent_activation='tanh', return_sequences=True, kernel_initializer=glorot_uniform(seed=self.seed), recurrent_initializer=Orthogonal(seed=self.seed)))
                    model.add(Dropout(self.dropout, seed=self.seed))
                else:
                    model.add(LSTM(self.cells_per_layer, stateful=self.stateful, recurrent_activation='tanh', return_sequences=True, dropout=self.dropout, recurrent_dropout=self.dropout))
                    model.add(Dropout(self.dropout))

        # ABBA requires softmax layer rather than dense layer with no activation.
        if isinstance(self.abba, ABBA):
            if verbose:
                print('\nAdded dense softmax layer and using categorical_crossentropy loss function! \n')
            if self.seed:
                model.add(Dense(self.features, kernel_initializer=glorot_uniform(seed=0)))
            else:
                model.add(Dense(self.features))
            model.add(Activation('softmax'))
        else:
            if verbose:
                print('\nAdded single neuron (no activation) and using mse loss function! \n')
            if self.seed:
                model.add(Dense(1, kernel_initializer=glorot_uniform(seed=0)))
            else:
                model.add(Dense(1))

        self.model = model
        if verbose:
            print('\nModel built! \n')


    def train(self, patience=100, max_epoch=100000, acceptable_loss=np.inf, verbose=True):
        """
        Train model on given time series.

        Parameter
        ---------
        patience - int
                Stopping criteria for training procedure. If the network experiences
                'patience' iterations with no improvement of the loss function then
                training stops, and the weights corresponding to smallest loss are
                used.

        max_epoch - int
                Maximum number of iterations through the training data during
                training process.

        acceptable_loss - float
                The maximum loss allowed before early stoppping criterion can be met.

        verbose - bool
                True - print progress
                False - print nothing
        """

        epoch = max_epoch
        self.patience = patience
        self.acceptable_loss = acceptable_loss

        # length of data required for prediction
        len_data = len(self.training_data)

        # Construct training arrays windows using timestep_back setting array to extract
        # arrays x and y.
        window = []
        if self.stateful:
            self.num_augs = min(self.l, len(self.training_data) - self.l)
            for el in range(self.num_augs):
                w = []
                for i in np.arange(el, len(self.training_data) - self.l, self.l):
                    w.append(self.training_data[i:i+self.l+1])
                window.append(np.array(w).astype(float))
        else:
            self.num_augs = 1
            w = []
            for i in np.arange(0, len(self.training_data) - self.l, 1):
                w.append(self.training_data[i:i+self.l+1])
            window.append(np.array(w).astype(float))

        # batch input of size (number of sequences, timesteps, data dimension)
        x = []
        for w in window:
            x.append(np.array(w[:, 0:-1]).reshape(-1, self.l, self.features))

        y = []
        for w in window:
            # Unable to generalise y for both numeric and symbolic data
            if isinstance(self.abba, ABBA):
                y.append(np.array(w[:, -1, :]))
            else:
                y.append(np.array(w[:, -1]))

        if isinstance(self.abba, ABBA):
            self.model.compile(loss='categorical_crossentropy', optimizer=Adam())
            if verbose:
                print('Feed ones through network:', self.model.evaluate(np.ones(shape=(1,self.l,self.features)), np.zeros(shape=(1,self.features)), batch_size=1, verbose=False))
        else:
            self.model.compile(loss='mse', optimizer=Adam())
            if verbose:
                print('Feed ones through network:', self.model.evaluate(np.ones(shape=(1,self.l,1)), np.zeros(shape=(1,1)), batch_size=1, verbose=False))

        weight_restarts = 10
        store_weights = [0]*weight_restarts
        initial_loss = [0]*weight_restarts
        for i in range(weight_restarts):
            if self.stateful:
                h = self.model.fit(x[0], y[0], epochs=1, batch_size=1, verbose=0, shuffle=False)
                initial_loss[i] = (h.history['loss'])[-1]
                self.model.reset_states()
                store_weights[i] = self.model.get_weights()

                # quick hack to reinitialise weights
                json_string = self.model.to_json()
                self.model = model_from_json(json_string)
                if isinstance(self.abba, ABBA):
                    self.model.compile(loss='categorical_crossentropy', optimizer=Adam())
                else:
                    self.model.compile(loss='mse', optimizer=Adam())
            else:
                h = self.model.fit(x[0], y[0], epochs=1, batch_size=1, verbose=0, shuffle=False) # no shuffling to remove randomness
                initial_loss[i] = (h.history['loss'])[-1]
                store_weights[i] = self.model.get_weights()
                self.model.reset_states()

                # quick hack to reinitialise weights
                json_string = self.model.to_json()
                self.model = model_from_json(json_string)
                if isinstance(self.abba, ABBA):
                    self.model.compile(loss='categorical_crossentropy', optimizer=Adam())
                else:
                    self.model.compile(loss='mse', optimizer=Adam())
        if verbose:
            print('Initial loss:', initial_loss)
        m = np.argmin(initial_loss)
        self.model.set_weights(store_weights[int(m)])
        del store_weights

        if verbose:
            print('\nTraining... \n')

        loss = np.zeros(epoch)
        min_loss = np.inf
        min_loss_ind = np.inf
        losses = [0]*self.num_augs
        if self.stateful: # no shuffle and reset state manually
            for iter in range(epoch):

                rint = np.random.permutation(self.num_augs)

                for r in rint:
                    h = self.model.fit(x[r], y[r], epochs=1, batch_size=1, verbose=0, shuffle=False)
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


    def start_prediction(self, randomize_abba=False):
        """
        Start prediction takes the first data points of the training data then
        makes a one step prediction. We then assume the one step prediction is true
        and continue to predict forward using the previously predicted data. We
        continue until we have constructed a time series the same length as the
        training data.

        Parameters
        ----------
        randomize_abba - bool
                When forecasting using ABBA representation, we can either
                forecast most likely symbol or include randomness in forecast.
                See jupyter notebook random_prediction_ABBA.ipynb.
        """

        model = self.model
        pred_l = self.l

        if isinstance(self.abba, ABBA):
            prediction_txt = self.ABBA_representation_string[0:pred_l]
            prediction = self.training_data[0:pred_l]
        else:
            prediction = self.training_data[0:pred_l].tolist()

        for ind in range(pred_l, len(self.training_data)):
            if self.stateful:
                window = []
                for i in np.arange(ind%pred_l, ind, pred_l):
                    window.append(prediction[i:i+pred_l])
            else:
                window = prediction[-pred_l:]

            pred_x =  np.array(window).astype(float)
            pred_x = np.array(pred_x).reshape(-1, pred_l, self.features)
            p = model.predict(pred_x, batch_size = 1)
            if isinstance(self.abba, ABBA):
                if randomize_abba:
                    # include some randomness in prediction
                    idx = np.random.choice(range(self.features), p=p[-1].ravel())
                else:
                    idx = np.argmax(p[-1], axis = 0)
                prediction_txt += self.alphabet[idx]
                add = np.zeros([1, self.features])
                add[0, idx] = 1
                prediction.append((add.tolist())[0])
            else:
                prediction.append(float(p[-1]))
            # reset states in case stateless
            model.reset_states()

        if isinstance(self.abba, ABBA):
            self.start_prediction_ts =  self.mean + np.dot(self.std,self.abba.inverse_transform(prediction_txt, self.centers, self.normalised_data[0]))
            self.start_prediction_txt = prediction_txt
        else:
            self.start_prediction_ts = self.mean + np.dot(self.std, prediction)


    def forecast_in_sample(self, randomize_abba=False):
        """
        In sample forecasting makes a one step prediction at every point of
        the training data. When ABBA is being used, this equates to one symbol
        symbol forecast, and so the plot will look a lot like multistep forecast
        for the numerical case.

        Parameters
        ----------
        randomize_abba - bool
                When forecasting using ABBA representation, we can either
                forecast most likely symbol or include randomness in forecast.
                See jupyter notebook random_prediction_ABBA.ipynb.
        """

        model = self.model
        pred_l = self.l

        prediction = []
        if isinstance(self.abba, ABBA):
            prediction_txt = []
        training_data = self.training_data[::]

        for ind in range(pred_l, len(self.training_data)):
            if self.stateful:
                window = []
                for i in np.arange(ind%pred_l, ind, pred_l):
                    window.append(training_data[i:i+pred_l])
            else:
                window = training_data[ind-pred_l:ind]

            window = np.array(window).astype(float)
            pred_x = np.array(window).reshape(-1,  pred_l, self.features)
            p = model.predict(pred_x, batch_size = 1)
            if isinstance(self.abba, ABBA):
                if randomize_abba:
                    # include some randomness in prediction
                    idx = np.random.choice(range(self.features), p=p[-1].ravel())
                else:
                    idx = np.argmax(p[-1], axis = 0)
                prediction_txt.append(self.alphabet[idx])

                tts = self.abba.inverse_transform(self.ABBA_representation_string[0:ind], self.centers, self.normalised_data[0])
                prediction += (self.abba.inverse_transform(self.alphabet[idx], self.centers, tts[-1]))

            else:
                prediction.append(float(p[-1]))
            # reset states in case stateless
            model.reset_states()

        if isinstance(self.abba, ABBA):
            self.in_sample_ts = self.mean + np.dot(self.std, prediction)
            self.in_sample_txt = prediction_txt
        else:
            self.in_sample_ts = self.mean +np.dot(self.std, prediction)


    def forecast_out_of_sample(self, l, randomize_abba=False, patches=True, remove_anomaly=True):
        """
        Given a fully trained LSTM model, forecast the next l subsequent datapoints.
        If ABBA representation has been used, this will forecast l symbols.

        Parameters
        ----------
        l - float
                Number of forecasted out_of_sample datapoints.

        randomize_abba - bool
                When forecasting using ABBA representation, we can either
                forecast most likely symbol or include randomness in forecast.
                See jupyter notebook random_prediction_ABBA.ipynb.

        patches - bool
                Use patches when creating forecasted time series. See ABBA module.

        remove_anomaly - bool
                Prevent forecast of any symbol which occurred only once during
                ABBA construction
        """
        if isinstance(self.abba, ABBA):
            prediction_txt = ''
            prediction = self.training_data[::]

            if remove_anomaly:
                c = dict(Counter(self.ABBA_representation_string))
                single_letters = [ord(key)-97 for key in c if c[key]==1]
        else:
            prediction = self.training_data[::].tolist()

        # Recursively make l one-step forecasts
        for ind in range(len(self.training_data), len(self.training_data) + l):

            # Build data to feed into model
            if self.stateful:
                window = []
                for i in np.arange(ind%self.l, ind, self.l):
                    window.append(prediction[i:i+self.l])
            else:
                window = prediction[-self.l:]
            pred_x =  np.array(window).astype(float)
            pred_x = np.array(pred_x).reshape(-1, self.l, self.features)

            # Feed through model
            p = self.model.predict(pred_x, batch_size = 1)

            # Convert to appropriate form
            if isinstance(self.abba, ABBA):
                if randomize_abba:
                    # include some randomness in prediction
                    if remove_anomaly:
                        distribution = p[-1].ravel()
                        distribution[single_letters] = 0 # remove probability form single letters
                        distribution /= sum(distribution) # scale so sum = 1
                        idx = np.random.choice(range(self.features), p=distribution)
                    else:
                        idx = np.random.choice(range(self.features), p=p[-1].ravel())
                else:
                    if remove_anomaly:
                        distribution = p[-1].ravel()
                        distribution[single_letters] = 0 # remove probability form single letters
                        idx = np.argmax(distribution, axis = 0)
                    else:
                        idx = np.argmax(p[-1], axis = 0)

                # Add forecast result to appropriate vectors.
                prediction_txt += self.alphabet[idx]
                add = np.zeros([1, self.features])
                add[0, idx] = 1
                prediction.append((add.tolist())[0])
            else:
                prediction.append(float(p[-1]))

            # reset states in case stateless
            self.model.reset_states()

        if isinstance(self.abba, ABBA):
            if patches:
                ABBA_patches = self.abba.get_patches(self.normalised_data, self.pieces, self.ABBA_representation_string, self.centers)
                # Construct mean of each patch
                d = {}
                for key in ABBA_patches:
                    d[key] = list(np.mean(ABBA_patches[key], axis=0))

                # Stitch patches together
                patched_ts = np.array([self.normalised_data[-1]])
                for letter in prediction_txt:
                    patch = d[letter]
                    patch -= patch[0] - patched_ts[-1] # shift vertically
                    patched_ts = np.hstack((patched_ts, patch[1:]))
                self.out_of_sample_ts =  self.mean + np.dot(self.std, patched_ts[1:])

            else:
                self.out_of_sample_ts =  self.mean + np.dot(self.std, self.abba.inverse_transform(prediction_txt, self.centers, self.normalised_data[0]))

            self.out_of_sample_txt = prediction_txt
        else:
            # Reverse normalisation procedure
            self.out_of_sample_ts = self.mean + np.dot(self.std, prediction[len(self.training_data):])


    def end_average_prediction(self, l, prob_int=0.1, randomize_abba=False):
        """
        Development function, not currently used.
        """

        if not isinstance(self.abba, ABBA):
            return None

        model = self.model
        pred_l = self.l

        prediction_txts = [self.ABBA_representation_string[::]]
        predictions = [self.training_data[::]]

        for ind in range(len(self.training_data), len(self.training_data) + l):
            # run through existing predictions
            for j in range(len(predictions)):
                if self.stateful:
                    window = []
                    for i in np.arange(ind%pred_l, ind, pred_l):
                        window.append(predictions[j][i:i+pred_l])
                else:
                    window = prediction[j][-pred_l:]

                pred_x =  np.array(window).astype(float)
                pred_x = np.array(pred_x).reshape(-1, pred_l, self.features)
                p = model.predict(pred_x, batch_size = 1)

                max_prob = np.max(p[-1], axis = 0)
                idxs = np.where(p[-1] > (max_prob-prob_int))

                for count, idx in enumerate(idxs[0]):
                    if count == 0:
                        prediction_txts[j] += self.alphabet[int(idx)]
                        add = np.zeros([1, self.features])
                        add[0, idx] = 1
                        predictions[j].append((add.tolist())[0])
                    else:
                        prediction_txts.append(prediction_txts[j]+self.alphabet[int(idx)])
                        add = np.zeros([1, self.features])
                        add[0, idx] = 1
                        predictions.append(predictions[j]+[(add.tolist())[0]])

                # reset states in case stateless
                model.reset_states()

        self.end_average_prediction_ts = []
        self.end_average_prediction_txt = prediction_txts
        for prediction_txt in prediction_txts:
            self.end_average_prediction_ts.append(self.mean + np.dot(self.std, self.abba.inverse_transform(prediction_txt, self.centers, self.normalised_data[0])))

        min_so_far = np.inf
        for ts in self.end_average_prediction_ts:
            if len(ts) < min_so_far:
                min_so_far = len(ts)

        mu = np.zeros(min_so_far)
        for ts in self.end_average_prediction_ts:
            mu += np.array(ts[0:min_so_far])

        mu = mu/len(self.end_average_prediction_ts)

        self.end_average_prediction_avg = mu


    def _figure(self, fig_ratio=.7, fig_scale=1):
        """
        Utility function for plot.
        """

        plt.figure(num=None, figsize=(5/fig_scale, 5*fig_ratio/fig_scale), dpi=80*fig_scale, facecolor='w', edgecolor='k')
        plt.subplots_adjust(left=0.11*fig_scale, right=1-0.05*fig_scale, bottom=0.085*fig_scale/fig_ratio, top=1-0.05*fig_scale/fig_ratio)


    def _savepdf(self, Activationfname='test', fig_scale=1):
        """
        Utility function for plot.
        """

        plt.savefig(fname+'.png', dpi=300*fig_scale, transparent=True)
        plt.savefig(fname+'.pdf', dpi=300*fig_scale, transparent=True)


    def plot(self, *, fname=None, type='end', fig_ratio=.7, fig_scale=1):
        """
        Simple plotting function for prediction.

        Parameters
        ----------
        fname - string
                Filename if saving plots. If fname = None, then plot is displayed
                rather than saved.

        type - string
                'start': Plot start prediction
                'point': Plot point prediction
                'end': Plot end prediction
        """

        if type == 'start':
            self._figure(fig_ratio, fig_scale)
            plot(self.ts)
            plot(self.start_prediction_ts, 'r')
            if fname == None:
                plt.show()
            else:
                self._savepdf(fname + '_' + 'start')

        elif type == 'point':
            self._figure(fig_ratio, fig_scale)
            plot(self.ts)
            plot(np.arange(self.l, self.l + len(self.point_prediction_ts)), self.point_prediction_ts, 'r')
            if fname == None:
                plt.show()
            else:
                self._savepdf(fname + '_' + 'point')

        elif type == 'end':
            self._figure(fig_ratio, fig_scale)
            plot(self.ts)
            plot(np.arange(len(self.ts)-1, len(self.end_prediction_ts)), self.end_prediction_ts[len(self.ts)-1:] , 'r')
            if fname == None:
                plt.show()
            else:
                self._savepdf(fname + '_' + 'end')

        else:
            raise TypeError('Type not recognised!')
