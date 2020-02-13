import numpy as np
import torch

class VanillaLSTM_pytorch(object):
    """ Vanilla LSTM implementation using pytorch """

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
            np.random.seed(seed)
            torch.manual_seed(seed)

    def build(self, sequence):
        """
        Build model
        """

        self.sequence = sequence

        # Sequence either list of lists or a list.
        if isinstance(sequence[0], type([])):
            self.features = len(sequence[0])
        else:
            self.features = 1

        self.model = pytorch_LSTM(input_dim=self.features, hidden_dim=self.cells_per_layer, batch_size=1, output_dim=self.features, num_layers=self.num_layers, dropout=self.dropout)
        if self.features == 1:
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters())
        else:
            self.loss_fn = torch.nn.MSELoss(size_average=False)
            self.optimizer = torch.optim.Adam(self.model.parameters())
        self.model.init_weights(self.model)

    def construct_training_data(self, other):
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
                    w.append(self.training_data[i:i+self.lag+1])
                window.append(np.array(w).astype(float))
        else:
            self.num_augs = 1
            w = []
            for i in np.arange(0, n - self.lag, 1):
                w.append(self.training_data[i:i+self.lag+1])
            window.append(np.array(w).astype(float))

        # batch input of size (number of sequences, timesteps, data dimension)
        x = []
        for w in window:
            x.append(np.array(w[:, 0:-1]).reshape(-1, self.lag, self.features))

        y = []
        for w in window:
            # Unable to generalise y for both numeric and symbolic data
            if self.features == 1:
                y.append(np.array(w[:, -1, :]))
            else:
                y.append(np.array(w[:, -1]).reshape(-1, 1))

        self.x = [torch.FloatTensor(xi) for xi in x]
        self.y = [torch.FloatTensor(yi) for yi in y]

    def train(self, other):
        """
        Train the model on the constructed training data
        """
        # Weight restarts
        states = self.model.initialise_states()
        weight_restarts = 10
        store_weights = [0]*weight_restarts
        initial_loss = [0]*weight_restarts
        for i in range(weight_restarts):
            # reset cell state
            states = self.model.initialise_states()
            y_pred, states = self.model(self.x[0][0], (states[0].detach(), states[1].detach()))

            # calculate loss
            if self.features == 1:
                self.loss = self.loss_fn(y_pred, self.y[0][0])
            else:
                target = torch.tensor([np.argmax(self.y[0][0], axis = 0)])
                self.loss = self.loss_fn(y_pred.reshape(1,-1), target)

            initial_loss[i] = self.loss.data
            store_weights[i] = self.model.state_dict()

            # Re initialise weights
            self.model.init_weights(self.model)
        m = np.argmin(initial_loss)
        self.model.load_state_dict(store_weights[int(m)])
        del store_weights

        # Train
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
                        y_pred, states = self.model(self.x[r][i], (states[0].detach(), states[1].detach()))

                        # calculate loss
                        if self.features == 1:
                            self.loss = self.loss_fn(y_pred, self.y[r][i])
                        else:
                            target = torch.tensor([np.argmax(self.y[r][i], axis = 0)])
                            self.loss = self.loss_fn(y_pred.reshape(1,-1), target)

                        loss_sum += (float(self.loss.data))**2

                        # Backward pass
                        self.loss.backward(retain_graph=True)

                        # Update parameters
                        self.optimizer.step()
                        # clear gradients
                        self.model.zero_grad()

                    losses[r] = loss_sum/self.x[r].shape[0]
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
                for i in np.random.permutation(self.x[0].shape[0]):

                    states = self.model.initialise_states()

                    # Forward pass
                    y_pred, states = self.model.forward(self.x[0][i], (states[0].detach(), states[1].detach()))

                    # calculate loss
                    if self.features == 1:
                        self.loss = self.loss_fn(y_pred, self.y[0][i])
                    else:
                        target = torch.tensor([np.argmax(self.y[0][i], axis = 0)])
                        self.loss = self.loss_fn(y_pred.reshape(1,-1), target)

                    loss_sum += (float(self.loss.data))**2

                    # Backward pass
                    self.loss.backward()

                    # Update parameters
                    self.optimizer.step()
                    # clear gradients
                    self.model.zero_grad()

                vec_loss[iter] = loss_sum/self.x[0].shape[0]

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


    def forecast(self, k, other):
        """
        Make k step forecast into the future.
        """
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


################################################################################
################################################################################
################################################################################


class pytorch_LSTM(torch.nn.Module):
    """
    Class to define LSTM model using pytorch.
    """
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers, dropout):
        super(pytorch_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.states = (0, 0)

        self.lstm = torch.nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=dropout)

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
