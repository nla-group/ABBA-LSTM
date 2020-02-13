# Check ABBA is available.
import importlib
import warnings
spec = importlib.util.find_spec("ABBA")
if spec is None:
    warnings.warn("Try: pip install -r 'requirements.txt'")
from ABBA import ABBA as ABBA

# Import modules
import torch
import numpy as np
from collections import Counter
import keras


class LSTM_model(object):
    """
    LSTM_model class used to build and train networks of LSTM cells for time
    series prediction using numerical data and symbolic data via ABBA compression.

    Possible attributes
    -------------------
    backend                         - either 'keras' or 'pytorch'
    verbose                         - True - print progress
                                      False - print nothing
                                      Note, this does not supress printing from
                                      ABBA module, must supress when constructing
                                      ABBA class object.
    model                           - network model
    num_layers                      - number of layers of stacked LSTMs
    cells_per_layer                 - number of LSTMs per layer
    dropout                         - amount of dropout after each LSTM layer
    abba                            - If no abba object is given, then time series
                                      prediction uses numerical representation.
                                      Otherwise abba class is used for symbolic
                                      conversion and network is trainined on the
                                      symbolic data.
    seed                            - random number generator seed

    ts                              - original time series
    normalised_data                 - z-normalised version of ts
    mean                            - mean of ts
    std                             - standard deviation of ts
    ABBA_representation_symbolic    - ABBA string representation of ts
    ABBA_representation_numeric     - ABBA numeric representation
    centers                         - cluster centers from ABBA compression
    alphabet                        - alphabet used in ABBA compression
    features                        - size of the alphabet

    training_data                   - data used for training network
    stateful                        - bool for 'stateful' or 'stateless training
                                      procedure. When 'stateful', the cell state
                                      information is passed between elements of
                                      the training set. The cell state must be
                                      manually reset and the order of the dataset
                                      is important.
    l                               - lag parameter, i.e. timesteps back controls
                                      the length of the 'short term' memory. The
                                      number of recurrent steps when training.

    patience (int)                  - Stopping criteria for training procedure.
                                      If the network experiences 'patience'
                                      iterations with no improvement of the loss
                                      function then training stops, and the weights
                                      corresponding to smallest loss are used.
    max_epoch (int)                 - Maximum number of iterations through the
                                      training data during training process.
    acceptable_loss (float)         - acceptable loss for training
    epoch                           - number of iterations used during training
    loss                            - list of loss value at each iteration

    generate_ts                     - numeric generative forecast
    generate_txt                    - symbolic generative forecast
    forecast_ts                     - numeric forecast
    forecast_txt                    - symbolic forecast
    """


    def __init__(self, num_layers=2, cells_per_layer=50, dropout=0.5, seed=None, backend='pytorch'):
        """
        Initialise class object. Read in shape of model.
        """
        self.num_layers = num_layers
        self.cells_per_layer = cells_per_layer
        self.dropout = dropout
        self.seed = seed
        self.backend = backend

        if seed != None and backend='pytorch':
            np.random.seed(seed)
            torch.manual_seed(seed)
        else:
            warnings.warn('Unable to seed when using keras backend')


    def build(self, ts, l=1, stateful=True, abba=None, verbose=True):
        """
        Build model, this function requires the time series and abba class object
        to understand input dimensions of the network.
        """

        # Read in parameters
        self.ts = ts
        self.abba = abba
        self.l = l
        self.stateful = stateful

        # Normalise time series
        self.mean = np.mean(ts)
        self.std = np.std(ts)
        self.normalised_data = (ts - self.mean)/self.std if self.std!=0 else ts - self.mean

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
        if self.backend = 'keras':
            self.model = build_Keras_LSTM(self.num_layers, self.cells_per_layer, self.l, self.features, self.stateful, self.seed)

        elif self.backend = 'pytorch':
            self.model = pytorch_LSTM(input_dim=self.features, hidden_dim=self.cells_per_layer, batch_size=1, output_dim=self.features, num_layers=self.num_layers, dropout=self.dropout)
            if isinstance(self.abba, ABBA):
                self.loss_fn = torch.nn.CrossEntropyLoss()
                self.optimizer = torch.optim.Adam(self.model.parameters())
            else:
                self.loss_fn = torch.nn.MSELoss(size_average=False)
                self.optimizer = torch.optim.Adam(self.model.parameters())
            self.model.init_weights(self.model)
        else:
            warning.warn('backend must be either keras or pytorch')

        if verbose:
            print('\nModel built! \n')

## UP TO HERE

    def train(self, patience=100, max_epoch=100000, acceptable_loss=np.inf, verbose=True):
        """
        Train model on given time series.
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
                #print(iter)
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
                #print(iter)
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
        self.epoch = iter+1
        self.loss = vec_loss[0:iter+1]


    def forecast_generate(self, randomize_abba=False, patches=True, remove_anomaly=True):
        """
        Generative forecasting takes the first l elements of the training data
        then iteratively forecasts to produce a time series of similar length to
        the training data.

        Parameters
        ----------
        randomize_abba - bool
                When predicting using abba representation, we can either predict most
                likely symbol or include randomness in prediction. See jupyter notebook
                random_prediction_ABBA.ipynb.

        patches - bool
                Use patches when creating forecasted time series. See ABBA module.

        remove_anomaly - bool
                Prevent forecast of any symbol which occurred only once during
                ABBA construction
        """
        # No more training, evaluate mode.
        self.model.eval()

        # Store first l values.
        if isinstance(self.abba, ABBA):
            prediction_txt = self.ABBA_representation_string[0:self.l]
            prediction = self.training_data[0:self.l]
            # Store single letters.
            if remove_anomaly:
                c = dict(Counter(self.ABBA_representation_string))
                single_letters = [ord(key)-97 for key in c if c[key]==1]
        else:
            prediction = self.training_data[0:self.l].tolist()

        # Recursively make one-step forecasts
        for ind in range(self.l, len(self.training_data)):

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
            states = self.model.initialise_states()
            for el in pred_x:
                p, states = self.model.forward(torch.tensor(el).float(), (states[0].detach(), states[1].detach()))

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
                patched_ts = np.array([self.normalised_data[0]])
                for letter in prediction_txt:
                    patch = d[letter]
                    patch -= patch[0] - patched_ts[-1] # shift vertically
                    patched_ts = np.hstack((patched_ts, patch[1:]))
                self.generate_ts =  self.mean + np.dot(self.std, patched_ts[1:])
            else:
                self.generate_ts =  self.mean + np.dot(self.std, self.abba.inverse_transform(prediction_txt, self.centers, self.normalised_data[0]))

            self.generate_txt = prediction_txt
        else:
            # Reverse normalisation procedure
            self.generate_ts = self.mean + np.dot(self.std, prediction)

    def forecast(self, fl, randomize_abba=False, patches=True, remove_anomaly=True):
        """
        Given a fully trained LSTM model, forecast the next fl subsequent datapoints.
        If ABBA representation has been used, this will forecast fl symbols.

        Parameters
        ----------
        fl - float
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
        self.model.eval()

        if isinstance(self.abba, ABBA):
            prediction_txt = ''
            prediction = self.training_data[::]

            if remove_anomaly:
                c = dict(Counter(self.ABBA_representation_string))
                single_letters = [ord(key)-97 for key in c if c[key]==1]
        else:
            prediction = self.training_data[::].tolist()

        # Recursively make fl one-step forecasts
        for ind in range(len(self.training_data), len(self.training_data) + fl):

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
            states = self.model.initialise_states()
            for el in pred_x:
                p, states = self.model.forward(torch.tensor(el).float(), (states[0].detach(), states[1].detach()))

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
