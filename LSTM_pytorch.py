# Check abba available to see if pip installed requirements.txt
import importlib
import warnings
spec = importlib.util.find_spec("ABBA")
if spec is None:
    warning.warn("Try: pip install -r 'requirements.txt'")
from ABBA import ABBA as ABBA

# import all other modules
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
from matplotlib.pyplot import plot,title,xlabel,ylabel,legend,grid,style,xlim,ylim,axis,show
from collections import Counter

class pytorch_LSTM(torch.nn.Module):
    """
    Class to define LSTM model using pytorch.
    """
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers, dropout, symbolic = False):
        super(pytorch_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.states = (0, 0)

        self.lstm = torch.nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=dropout)

        if symbolic:
            self.dropout = torch.nn.Dropout(p=dropout)
            self.final = torch.nn.Linear(self.hidden_dim, self.output_dim)
        else:
            self.dropout = torch.nn.Dropout(p=dropout)
            self.final = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def init_weights(self, m):
        for n, p in m.named_parameters():
            if 'weight_ih' in n:
                for ih in p.chunk(4,0):
                    torch.nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in n:
                for hh in p.chunk(4,0):
                    torch.nn.init.orthogonal_(hh)
            elif 'bias_ih' in n:
                torch.nn.init.zeros_(p)
            elif 'bias_hh' in n:
                torch.nn.init.zeros_(p)
            elif 'final.weight' in n:
                torch.nn.init.xavier_uniform_(p)
            elif 'final.bias' in n:
                torch.nn.init.zeros_(p)

    def initialise_states(self):
        """
        Reset both cell state and hidden state
        """
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim), torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def reset_hidden(self, states):
        """
        Reset hidden state
        """
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim), states[1])

    def reset_cell(self):
        """
        Reset cell state
        """
        return (states[0], torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input, states):
        """
        Define forward pass through LSTM
        """
        # pass through LSTM layers
        lstm_out, states = self.lstm(input.view(len(input), self.batch_size, -1), states)
        # pass through linear layer
        y_pred = self.final(self.dropout(lstm_out[-1].view(self.batch_size, -1)))
        return y_pred.view(-1), states


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
                Seed for weight initialisation and augmentation choice.
        """

        self.num_layers = num_layers
        self.cells_per_layer = cells_per_layer
        self.dropout = dropout
        self.seed = seed

        if seed != None:
            np.random.seed(seed)
            torch.manual_seed(seed)


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
                reset and the order of the dataset is important. Recommend using
                augmentation in the training procedure.

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
        if isinstance(self.abba, ABBA):
            if verbose:
                print('\nAdded dense softmax layer and using categorical_crossentropy loss function! \n')
            self.model = pytorch_LSTM(input_dim=self.features, hidden_dim=self.cells_per_layer, batch_size=1, output_dim=self.features, num_layers=self.num_layers, dropout=self.dropout, symbolic=True)
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters())

        else:
            if verbose:
                print('\nAdded single neuron (no activation) and using mse loss function! \n')
            self.model = pytorch_LSTM(input_dim=self.features, hidden_dim=self.cells_per_layer, batch_size=1, output_dim=1, num_layers=self.num_layers, dropout=self.dropout, symbolic=False)
            self.loss_fn = torch.nn.MSELoss(size_average=False)
            self.optimizer = torch.optim.Adam(self.model.parameters())

        if verbose:
            print('\nModel built! \n')

        self.model.init_weights(self.model)


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
                y.append(np.array(w[:, -1]).reshape(-1, self.features))

        x = [torch.FloatTensor(xi) for xi in x]
        y = [torch.FloatTensor(yi) for yi in y]

        states = self.model.initialise_states()

        if isinstance(self.abba, ABBA):
            if verbose:
                print('Sanity check: feed ones through network:', self.model(torch.tensor(np.ones(shape=(self.l,self.features))).float(), states)[0])
        else:
            if verbose:
                print('Sanity check: feed ones through network:', self.model(torch.tensor(np.ones(shape=(self.l,self.features))).float(), states)[0])

        # Try weight restarts
        weight_restarts = 10
        store_weights = [0]*weight_restarts
        initial_loss = [0]*weight_restarts
        for i in range(weight_restarts):
            # reset cell state
            states = self.model.initialise_states()
            y_pred, states = self.model(x[0][0], (states[0].detach(), states[1].detach()))

            # calculate loss
            if self.features == 1:
                self.loss = self.loss_fn(y_pred, y[0][0])
            else:
                target = torch.tensor([np.argmax(y[0][0], axis = 0)])
                self.loss = self.loss_fn(y_pred.reshape(1,-1), target)

            initial_loss[i] = self.loss.data
            store_weights[i] = self.model.state_dict()

            # Re initialise weights
            self.model.init_weights(self.model)

        if verbose:
            print('Initial loss:', initial_loss)
        m = np.argmin(initial_loss)
        self.model.load_state_dict(store_weights[int(m)])
        del store_weights

        if verbose:
            print('\nTraining... \n')

        vec_loss = np.zeros(epoch)
        min_loss = np.inf
        min_loss_ind = np.inf
        losses = [0]*self.num_augs
        if self.stateful: # no shuffle and reset state manually
            for iter in range(epoch):
                rint = np.random.permutation(self.num_augs)

                for r in rint:
                    # reset cell state
                    states = self.model.initialise_states()

                    loss_sum = 0
                    for i in range(x[r].shape[0]):
                        # Forward pass
                        y_pred, states = self.model(x[r][i], (states[0].detach(), states[1].detach()))

                        # calculate loss
                        if self.features == 1:
                            self.loss = self.loss_fn(y_pred, y[r][i])
                        else:
                            target = torch.tensor([np.argmax(y[r][i], axis = 0)])
                            self.loss = self.loss_fn(y_pred.reshape(1,-1), target)

                        loss_sum += (float(self.loss.data))**2

                        # Backward pass
                        self.loss.backward(retain_graph=True)

                        # Update parameters
                        self.optimizer.step()
                        # clear gradients
                        self.model.zero_grad()

                    losses[r] = loss_sum/x[r].shape[0]
                vec_loss[iter] = np.mean(losses)

                if vec_loss[iter] >= min_loss:
                    if iter%100 == 0 and verbose:
                        print('iteration:', iter)
                    if iter - min_loss_ind >= self.patience and vec_loss[iter]<self.acceptable_loss:
                        break
                else:
                    min_loss = vec_loss[iter]
                    old_weights = self.model.state_dict()
                    if verbose:
                        print('iteration:', iter, 'loss:', min_loss)
                    min_loss_ind = iter
            if verbose:
                print('iteration:', iter)

        else: # shuffle in fit
            for iter in range(epoch):
                loss_sum = 0
                for i in np.random.permutation(x[0].shape[0]):

                    states = self.model.initialise_states()

                    # Forward pass
                    y_pred, states = self.model.forward(x[0][i], (states[0].detach(), states[1].detach()))

                    # calculate loss
                    if self.features == 1:
                        self.loss = self.loss_fn(y_pred, y[0][i])
                    else:
                        target = torch.tensor([np.argmax(y[0][i], axis = 0)])
                        self.loss = self.loss_fn(y_pred.reshape(1,-1), target)

                    loss_sum += (float(self.loss.data))**2

                    # Backward pass
                    self.loss.backward()

                    # Update parameters
                    self.optimizer.step()
                    # clear gradients
                    self.model.zero_grad()

                vec_loss[iter] = loss_sum/x[0].shape[0]

                if vec_loss[iter] >= min_loss:
                    if iter%100 == 0 and verbose:
                        print('iteration:', iter)
                    if iter - min_loss_ind >= self.patience and vec_loss[iter]<self.acceptable_loss:
                        break
                else:
                    min_loss = vec_loss[iter]
                    old_weights = self.model.state_dict()
                    if verbose:
                        print('iteration:', iter, 'loss:', min_loss)
                    min_loss_ind = iter
            if verbose:
                print('iteration:', iter)

        if verbose:
            print('\nTraining complete! \n')

        self.model.load_state_dict(old_weights)
        self.epoch = iter
        self.loss = vec_loss[0:iter]


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
                When predicting using abba representation, we can either predict most
                likely symbol or include randomness in prediction. See jupyter notebook
                random_prediction_ABBA.ipynb.
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

            states = model.initialise_states()
            for el in pred_x:
                p, states = model.forward(torch.tensor(el).float(), (states[0].detach(), states[1].detach()))

            if isinstance(self.abba, ABBA):
                softmax = torch.nn.Softmax(dim=-1)
                p = softmax(p).tolist()
                p = np.array(p)
                p /= p.sum()
                if randomize_abba:
                    # include some randomness in prediction
                    idx = np.random.choice(range(self.features), p=p.ravel())
                else:
                    idx = np.argmax(list(p), axis = 0)
                prediction_txt += self.alphabet[idx]
                add = np.zeros([1, self.features])
                add[0, idx] = 1
                prediction.append((add.tolist())[0])
            else:
                prediction.append(float(p))

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
        model.eval()
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

            states = model.initialise_states()
            for el in pred_x:
                p, states = model.forward(torch.tensor(el).float(), (states[0].detach(), states[1].detach()))
                softmax = torch.nn.Softmax(dim=-1)
                p = softmax(p).tolist()
                p = np.array(p)
                p /= p.sum()
                if randomize_abba:
                    # include some randomness in prediction
                    idx = np.random.choice(range(self.features), p=(p.ravel()))
                else:
                    idx = np.argmax(list(p), axis = 0)
                prediction_txt.append(self.alphabet[idx])

                tts = self.abba.inverse_transform(self.ABBA_representation_string[0:ind], self.centers, self.normalised_data[0])
                prediction += (self.abba.inverse_transform(self.alphabet[idx], self.centers, tts[-1]))

            else:
                prediction.append(float(p))

        if isinstance(self.abba, ABBA):
            self.point_prediction_ts = self.mean + np.dot(self.std, prediction)
            self.point_prediction_txt = prediction_txt
        else:
            self.point_prediction_ts = self.mean +np.dot(self.std, prediction)


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

        model = self.model
        model.eval()
        pred_l = self.l


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
                for i in np.arange(ind%pred_l, ind, pred_l):
                    window.append(prediction[i:i+pred_l])
            else:
                window = prediction[-pred_l:]
            pred_x =  np.array(window).astype(float)
            pred_x = np.array(pred_x).reshape(-1, pred_l, self.features)

            # Feed through model
            states = model.initialise_states()
            for el in pred_x:
                p, states = model.forward(torch.tensor(el).float(), (states[0].detach(), states[1].detach()))

            # Convert to appropriate form
            if isinstance(self.abba, ABBA):
                softmax = torch.nn.Softmax(dim=-1)
                p = softmax(p).tolist()
                p = np.array(p)
                p /= p.sum()
                if randomize_abba:
                    # include some randomness in prediction
                    if remove_anomaly:
                        distribution = p
                        distribution[single_letters] = 0 # remove probability form single letters
                        distribution /= sum(distribution) # scale so sum = 1
                        idx = np.random.choice(range(self.features), p=distribution)
                    else:
                        idx = np.random.choice(range(self.features), p=(p.ravel()))
                else:
                    if remove_anomaly:
                        distribution = p
                        distribution[single_letters] = 0 # remove probability form single letters
                        idx = np.argmax(distribution, axis = 0)
                    else:
                        idx = np.argmax(list(p), axis = 0)

                # Add forecast result to appropriate vectors.
                prediction_txt += self.alphabet[idx]
                add = np.zeros([1, self.features])
                add[0, idx] = 1
                prediction.append((add.tolist())[0])
            else:
                prediction.append(float(p))

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


    def plot(self, fname=None, type='end', fig_ratio=.7, fig_scale=1):
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
