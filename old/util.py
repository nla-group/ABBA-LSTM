import torch
import keras as K

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


def build_Keras_LSTM(num_layers, cells_per_layer, l, features, stateful, seed):
    model = K.models.Sequential()
    for index in range(num_layers):
        if index == 0:
            if num_layers == 1:
                if seed:
                    model.add(K.layers.LSTM(cells_per_layer, batch_input_shape=(1, l, features), recurrent_activation='tanh', stateful=stateful, return_sequences=False, kernel_initializer=K.initializers.glorot_uniform(seed=seed), recurrent_initializer=K.initializers.Orthogonal(seed=seed)))
                    model.add(K.layers.Dropout(dropout, seed=seed))
                else:
                    model.add(K.layers.LSTM(cells_per_layer, batch_input_shape=(1, l, features), recurrent_activation='tanh', stateful=stateful, return_sequences=False))
                    model.add(K.layers.Dropout(dropout))
            else:
                if seed:
                    model.add(K.layers.LSTM(cells_per_layer, batch_input_shape=(1, l, features), recurrent_activation='tanh', stateful=stateful, return_sequences=True, kernel_initializer=K.initializers.glorot_uniform(seed=seed), recurrent_initializer=K.initializers.Orthogonal(seed=seed)))
                    model.add(K.layers.Dropout(dropout, seed=seed))
                else:
                    model.add(K.layers.LSTM(cells_per_layer, batch_input_shape=(1, l, features), recurrent_activation='tanh', stateful=stateful, return_sequences=True))
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
