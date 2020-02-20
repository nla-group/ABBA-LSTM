import numpy as np
import torch

class batchless_VanillaLSTM_pytorch(object):
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

    def build(self, sequence, debug=False):
        """
        Build model
        """
        # Sequence either list of lists or a list.
        if sequence.ndim != 1:
            self.features = len(sequence[0])
        else:
            self.features = 1

        # Reshape and convert to torch tensor
        self.sequence = torch.FloatTensor(sequence).view(-1, 1, self.features)

        self.model = pytorch_LSTM(input_dim=self.features, hidden_dim=self.cells_per_layer, batch_size=1, output_dim=self.features, num_layers=self.num_layers, dropout=self.dropout)
        if self.features != 1:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
            self.optimizer = torch.optim.Adam(self.model.parameters())
        else:
            self.loss_fn = torch.nn.MSELoss(size_average=False)
            self.optimizer = torch.optim.Adam(self.model.parameters())
        self.model.init_weights(self.model)

    def construct_training_index(self, debug=False):
        """
        Construct training index (compatible with model) from sequence of vectors of dimension d,
        """
        n = len(self.sequence)
        self.index = []
        if self.stateful:
            # Create groups
            self.num_augs = min(self.lag, n - self.lag)
            for el in range(self.num_augs):
                self.index.append(np.arange(el, n - self.lag, self.lag))
        else:
            self.num_augs = 1
            self.index = np.arange(0, n - self.lag, 1)

    def train(self, patience=10, max_epoch=1000, acceptable_loss=np.inf, weight_restarts=False, debug=False):
        """
        Train the model on the constructed training data
        """
        ########################################################################
        # Weight restarts
        ########################################################################
        states = self.model.initialise_states()
        if weight_restarts:
            weight_restarts = 10
            store_weights = [0]*weight_restarts
            initial_loss = [0]*weight_restarts
            for i in range(weight_restarts):
                # reset cell state
                states = self.model.initialise_states()

                y_pred, states = self.model(self.sequence[0:self.lag, :, :], (states[0].detach(), states[1].detach()))

                # calculate loss
                if self.features == 1:
                    self.loss = self.loss_fn(y_pred.view(-1, 1), self.sequence[self.lag, :, :])
                else:
                    target = self.sequence[self.lag, :, :].max(-1)[1]
                    self.loss = self.loss_fn(y_pred.reshape(1,-1), target)

                initial_loss[i] = self.loss.data
                store_weights[i] = self.model.state_dict()

                # Re initialise weights
                self.model.init_weights(self.model)
            m = np.argmin(initial_loss)
            self.model.load_state_dict(store_weights[int(m)])
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
                rint = np.random.permutation(self.num_augs) # shuffle groups
                for r in rint: # run through groups
                    # reset cell state
                    states = self.model.initialise_states()

                    loss_sum = 0
                    for i in self.index[r]: # run through group
                        # Forward pass
                        y_pred, states = self.model(self.sequence[i:i+self.lag, :, :], (states[0].detach(), states[1].detach()))

                        # calculate loss
                        if self.features == 1:
                            self.loss = self.loss_fn(y_pred.view(-1, 1), self.sequence[i+self.lag, :, :])
                        else:
                            target = self.sequence[i+self.lag, :, :].max(-1)[1]
                            self.loss = self.loss_fn(y_pred.reshape(1,-1), target)

                        loss_sum += (float(self.loss.data))**2

                        # Backward pass
                        self.loss.backward(retain_graph=True)

                        # Update parameters
                        self.optimizer.step()
                        # clear gradients
                        self.model.zero_grad()

                    losses[r] = loss_sum/len(self.index[r])
                vec_loss[iter] = np.mean(losses)

                if vec_loss[iter] >= min_loss:
                    if iter - min_loss_ind >= patience and min_loss<acceptable_loss:
                        break
                else:
                    min_loss = vec_loss[iter]
                    old_weights = self.model.state_dict()
                    min_loss_ind = iter

        else: # shuffle in fit
            for iter in range(max_epoch):
                loss_sum = 0
                for i in np.random.permutation(len(self.index)):
                    states = self.model.initialise_states()

                    # Forward pass
                    y_pred, states = self.model.forward(self.sequence[i:i+self.lag, :, :], (states[0].detach(), states[1].detach()))

                    # calculate loss
                    if self.features == 1:
                        self.loss = self.loss_fn(y_pred.view(-1, 1), self.sequence[i+self.lag, :, :])
                    else:
                        target = self.sequence[i+self.lag, :, :].max(-1)[1]
                        self.loss = self.loss_fn(y_pred.reshape(1,-1), target)

                    loss_sum += (float(self.loss.data))**2

                    # Backward pass
                    self.loss.backward()

                    # Update parameters
                    self.optimizer.step()
                    # clear gradients
                    self.model.zero_grad()

                vec_loss[iter] = loss_sum/len(self.index)

                if vec_loss[iter] >= min_loss:
                    if iter - min_loss_ind >= patience and min_loss < acceptable_loss:
                        break
                else:
                    min_loss = vec_loss[iter]
                    old_weights = self.model.state_dict()
                    min_loss_ind = iter

        self.model.load_state_dict(old_weights)
        self.epoch = iter+1
        self.loss = vec_loss[0:iter+1]


    def forecast(self, k, randomize=False, debug=False):
        """
        Make k step forecast into the future.
        """
        self.model.eval()
        prediction = self.sequence.clone()

        # Recursively make k one-step forecasts
        for ind in range(len(self.sequence), len(self.sequence) + k):
            # Build data to feed into model
            if self.stateful:
                index = np.arange(ind%self.lag, ind, self.lag)
            else:
                index = [ind - self.lag]

            # Feed through model
            states = self.model.initialise_states()
            for i in index:
                p, states = self.model.forward(prediction[i:i+self.lag, :, :], (states[0].detach(), states[1].detach()))

            # Convert output
            if self.features != 1:
                softmax = torch.nn.Softmax(dim=-1)
                p = softmax(p).tolist()
                p = np.array(p)
                p /= p.sum()
                if randomize:
                    idx = np.random.choice(range(self.features), p=(p.ravel()))
                else:
                    idx = np.argmax(list(p), axis = 0)

                # Add forecast result to appropriate vectors.
                pred = torch.zeros([1, 1, self.features])
                pred[0, 0, idx] = 1
            else:
                pred = torch.zeros([1, 1, 1])
                pred[0, 0, 0] = p

            prediction = torch.cat([prediction, pred], dim=0)

        if self.features != 1:
            return prediction.view(-1, self.features).tolist()
        else:
            return prediction.view(-1).detach()


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
