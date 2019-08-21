import numpy as np
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.initializers import Orthogonal, glorot_uniform
from keras.optimizers import Adam
import json
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('classic')
from matplotlib.pyplot import plot,title,xlabel,ylabel,legend,grid,style,xlim,ylim,axis,show
import sys
sys.path.append('./ABBA')
from ABBA import ABBA as ABBA
import os

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
    ABBA_representation_symbolic    - ABBA string representation of ts
    centers                         - cluster centers from ABBA compression
    ABBA_representation_numeric     - ABBA numeric representation
    alphabet                        - alphabet used in ABBA compression
    training_data                   - data used for training network
    features                        - size of the alphabet
    model                           - network model
    patience                        - patience parameter for training
    optimizer                       - optimization algorithm used
    epoch                           - number of iterations used during training
    loss                            - list of loss value at each iteration
    start_prediction_ts             - numeric start prediction
    start_prediction_txt            - symbolic start prediction
    point_prediction_ts             - numeric point prediction
    point_prediction_txt            - symbolic point prediction
    end_prediction_ts               - numeric end prediction
    end_prediction_txt              - symbolic end prediction
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
            # Force TensorFlow to use single thread.
            # Multiple threads are a potential source of non-reproducible results.
            # For further details, see: https://stackoverflow.com/questions/42022950/
            from keras import backend as k
            session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                          inter_op_parallelism_threads=1)
            tf.compat.v1.set_random_seed(seed)

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
        if isinstance(self.abba, ABBA.ABBA):
            if verbose:
                print('\nApplying ABBA compression! \n')
            # Apply abba transformation
            self.ABBA_representation_string, self.centers = abba.transform(self.normalised_data)
            self.ABBA_representation_numerical = abba.inverse_transform(self.ABBA_representation_string, self.centers, self.normalised_data[0])

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

        if isinstance(self.abba, ABBA.ABBA):
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


    def train(self, patience=100, max_epoch=100000, verbose=True):
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

        verbose - bool
                True - print progress
                False - print nothing
        """

        epoch = max_epoch
        self.patience = patience

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
            if isinstance(self.abba, ABBA.ABBA):
                y.append(np.array(w[:, -1, :]))
            else:
                y.append(np.array(w[:, -1]))

        if isinstance(self.abba, ABBA.ABBA):
            self.model.compile(loss='categorical_crossentropy', optimizer=Adam())
            if verbose:
                print('Feed ones through network:', self.model.evaluate(np.ones(shape=(1,self.l,self.features)), np.zeros(shape=(1,self.features)), batch_size=1, verbose=False))
        else:
            self.model.compile(loss='mse', optimizer=Adam())
            if verbose:
                print('Feed ones through network:', self.model.evaluate(np.ones(shape=(1,self.l,1)), np.zeros(shape=(1,1)), batch_size=1, verbose=False))

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
                    if iter - min_loss_ind >= self.patience:
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
                    if iter - min_loss_ind >= self.patience:
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


    def start_prediction(self, use_recurrence=True, randomize_abba=False):
        """
        Start prediction takes the first data points of the training data then
        makes a one step prediction. We then assume the one step prediction is true
        and continue to predict forward using the previously predicted data. We
        continue until we have constructed a time series the same length as the
        training data.

        Parameters
        ----------
        randomize_abba - bool
                When predicting using abba representation, we can either predict most
                likely symbol or include randomness in prediction. See jupyter notebook
                random_prediction_ABBA.ipynb.
        """

        model = self.model
        pred_l = self.l

        if isinstance(self.abba, ABBA.ABBA):
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
            if isinstance(self.abba, ABBA.ABBA):
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

        if isinstance(self.abba, ABBA.ABBA):
            self.start_prediction_ts =  self.mean + np.dot(self.std,self.abba.inverse_transform(prediction_txt, self.centers, self.normalised_data[0]))
            self.start_prediction_txt = prediction_txt
        else:
            self.start_prediction_ts = self.mean + np.dot(self.std, prediction)


    def point_prediction(self, use_recurrence=True, randomize_abba=False):
        """
        Point prediction consists of making a one step prediction at every point of
        the training data. When ABBA is being used, this equates to one symbol
        symbol prediction, and so the plot will look a lot like multistep prediction
        for the numerical case.

        Parameters
        ----------
        randomize_abba - bool
                When predicting using abba representation, we can either predict most
                likely symbol or include randomness in prediction. See jupyter notebook
                random_prediction_ABBA.ipynb.
        """

        model = self.model
        pred_l = self.l

        prediction = []
        if isinstance(self.abba, ABBA.ABBA):
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
            if isinstance(self.abba, ABBA.ABBA):
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

        if isinstance(self.abba, ABBA.ABBA):
            self.point_prediction_ts = self.mean + np.dot(self.std, prediction)
            self.point_prediction_txt = prediction_txt
        else:
            self.point_prediction_ts = self.mean +np.dot(self.std, prediction)


    def end_prediction(self, l, use_recurrence=True, randomize_abba=False):
        """
        We take the training data and then predict what happens next. This is clearly
        the most useful as we are predicting data the model has never seen. The other
        methods of prediction merely represent how well the LSTM has 'learnt' the
        behaviour of the time series.

        Parameters
        ----------
        randomize_abba - bool
                When predicting using abba representation, we can either predict most
                likely symbol or include randomness in prediction. See jupyter notebook
                random_prediction_ABBA.ipynb.
        """

        model = self.model
        pred_l = self.l

        if isinstance(self.abba, ABBA.ABBA):
            prediction_txt = self.ABBA_representation_string[::]
            prediction = self.training_data[::]
        else:
            prediction = self.training_data[::].tolist()

        for ind in range(len(self.training_data), len(self.training_data) + l):
            if self.stateful:
                window = []
                for i in np.arange(ind%pred_l, ind, pred_l):
                    window.append(prediction[i:i+pred_l])
            else:
                window = prediction[-pred_l:]

            pred_x =  np.array(window).astype(float)
            pred_x = np.array(pred_x).reshape(-1, pred_l, self.features)
            p = model.predict(pred_x, batch_size = 1)

            if isinstance(self.abba, ABBA.ABBA):
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

        if isinstance(self.abba, ABBA.ABBA):
            self.end_prediction_ts =  self.mean + np.dot(self.std, self.abba.inverse_transform(prediction_txt, self.centers, self.normalised_data[0]))
            self.end_prediction_txt = prediction_txt
        else:
            self.end_prediction_ts = self.mean + np.dot(self.std, prediction)


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
