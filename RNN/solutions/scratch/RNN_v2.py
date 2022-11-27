class RNN:
    # A Vanilla Recurrent Neural Network.

    def __init__(self, input_size, output_size, hidden_size=64):
        # Weights
        self.Whh = rd.randn(hidden_size, hidden_size) / 1000
        self.Wxh = rd.randn(hidden_size, input_size) / 1000
        self.Why = rd.randn(output_size, hidden_size) / 1000

        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    # ----- #
    
    def forward(self, inputs):
        '''
        Perform a forward pass of the RNN using the given inputs.
        Returns the final output and hidden state.
        - inputs is an array of one-hot vectors with shape (input_size, 1).
        '''
        h = np.zeros((self.Whh.shape[0], 1))

        self.inputs = inputs
        self.hs = { 0: h }
        
        # Perform each step of the RNN
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.hs[i + 1] = h
            
        # Compute the output
        y = self.Why @ h + self.by

        return y, h
    
    # ----- #
    
    def backprop(self, d_y, learn_rate=2e-2):
        '''    
        Perform a backward pass of the RNN.    
        - d_y (dL/dy) has shape (output_size, 1).    
        - learn_rate is a float.    
        '''    
        n = len(self.inputs)

        # Calculate dL/dWhy and dL/dby.
        d_Why = d_y @ self.hs[n].transpose()
        d_by = d_y
        
        # Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero.
        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        # Calculate dL/dh for the last h.
        d_h = self.Why.transpose() @ d_y

        # Backpropagate through time.
        for t in reversed(range(n)):
            # An intermediate value: dL/dh * (1 - h^2)
            tmp = (1 - self.hs[t+1]**2) * d_h

            # dL/db = dL/dh * (1 - h^2)
            d_bh += tmp
            # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
            d_Whh += tmp @ self.hs[t].transpose()
            # dL/dWxh = dL/dh * (1 - h^2) * x
            d_Wxh += tmp @ self.inputs[t].transpose()
            # Next dL/dh = dL/dh * (1 - h^2) * Whh
            d_h = self.Whh @ tmp
            
        # Clip to prevent exploding gradients.
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)
            
        # Update weights and biases using gradient descent.
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by