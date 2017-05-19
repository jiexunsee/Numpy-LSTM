import numpy as np


''' An LSTM with a FC layer on top '''
class LSTM:
    def sigmoid(self, x):
        return (1/(1+np.exp(-x)))

    def sigmoid_derivative(self, x):
        return (x*(1-x))

    def tanh_derivative(self, x):
        return (1-(np.tanh(x))**2)

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.hidden_to_out = np.random.uniform(-0.1, 0.1, size=(hidden_dim, output_dim))
        self.U_i = np.random.uniform(-0.1, 0.1, size=(input_dim, hidden_dim))
        self.U_f = np.random.uniform(-0.1, 0.1, size=(input_dim, hidden_dim))
        self.U_o = np.random.uniform(-0.1, 0.1, size=(input_dim, hidden_dim))
        self.U_g = np.random.uniform(-0.1, 0.1, size=(input_dim, hidden_dim))
        self.W_i = np.random.uniform(-0.1, 0.1, size=(hidden_dim, hidden_dim))
        self.W_f = np.random.uniform(-0.1, 0.1, size=(hidden_dim, hidden_dim))
        self.W_o = np.random.uniform(-0.1, 0.1, size=(hidden_dim, hidden_dim))
        self.W_g = np.random.uniform(-0.1, 0.1, size=(hidden_dim, hidden_dim))

        self.cell_state = np.zeros((1, hidden_dim))
        self.hidden_state = np.zeros((1, hidden_dim))

        self.input_gates = list()
        self.forget_gates = list()
        self.output_gates = list()
        self.gs = list() # g refers to the candidate cell state vector
        self.cell_states = list()
        self.hidden_states = list()
        self.outputs = list()

        self.del_predicts = list()
        self.del_hidden_states = list()
        self.del_cell_states = list()
        self.del_gs = list()
        self.del_input_gates = list()
        self.del_forget_gates = list()
        self.del_output_gates = list()

    def forward(self, x):
        input_gate = self.sigmoid((x @ self.U_i)+(self.hidden_state @ self.W_i)) # a short form for np.dot()
        # input_gate = self.sigmoid(np.dot(x, self.U_i) + np.dot(self.hidden_state, self.W_i))
        forget_gate = self.sigmoid(np.dot(x, self.U_f) + np.dot(self.hidden_state, self.W_f))
        output_gate = self.sigmoid(np.dot(x, self.U_o) + np.dot(self.hidden_state, self.W_o))
        g = np.tanh(np.dot(x, self.U_g) + np.dot(self.hidden_state, self.W_g))
        self.cell_state = self.cell_state*forget_gate + g*input_gate
        self.hidden_state = np.tanh(self.cell_state)*output_gate
        output = np.dot(self.hidden_state, self.hidden_to_out)
        output = self.sigmoid(output)

        self.input_gates.append(input_gate)
        self.forget_gates.append(forget_gate)
        self.output_gates.append(output_gate)
        self.gs.append(g)
        self.cell_states.append(self.cell_state)
        self.hidden_states.append(self.hidden_state)
        self.outputs.append(output)

    def backward(self, x, y, seq_length, lr):
        timestep_error = 0
        future_del_cell_state = 0

        for a in range(seq_length):
            if a == 0:
                future_forget_gate = np.zeros_like(self.forget_gates[0])
            else:
                future_forget_gate = self.forget_gates[seq_length-a]
            if a == seq_length:
                prev_cell_state = np.zeros_like(self.cell_states[0])
            else:
                prev_cell_state = self.cell_states[seq_length-a-2]

            del_predict = (self.outputs[seq_length-a-1] - y[seq_length-a-1])*(self.sigmoid_derivative(self.outputs[seq_length-a-1]))
            lstm_error = (self.hidden_to_out @ del_predict.T).T*self.sigmoid_derivative(self.hidden_states[seq_length-a-1])
            del_hidden_state = lstm_error + timestep_error
            del_cell_state = del_hidden_state*self.output_gates[seq_length-a-1]*self.tanh_derivative(self.cell_states[seq_length-a-1]) + future_del_cell_state*future_forget_gate
            del_g = del_cell_state*self.input_gates[seq_length-a-1]*self.tanh_derivative(self.gs[seq_length-a-1])
            del_input_gate = del_cell_state*self.gs[seq_length-a-1]*self.sigmoid_derivative(self.input_gates[seq_length-a-1])
            del_forget_gate = del_cell_state*prev_cell_state*self.sigmoid_derivative(self.forget_gates[seq_length-a-1])
            del_output_gate = del_hidden_state*np.tanh(self.cell_states[seq_length-a-1])*self.sigmoid_derivative(self.output_gates[seq_length-a-1])

            timestep_error = (np.dot(self.W_i, del_input_gate.T) + np.dot(self.W_f, del_forget_gate.T) + np.dot(self.W_o, del_output_gate.T) + np.dot(self.W_g, del_g.T)).T
            future_del_cell_state = del_cell_state

            self.del_predicts.append(del_predict)
            self.del_hidden_states.append(del_hidden_state)
            self.del_cell_states.append(del_cell_state)
            self.del_gs.append(del_g)
            self.del_input_gates.append(del_input_gate)
            self.del_forget_gates.append(del_forget_gate)
            self.del_output_gates.append(del_output_gate)

        #UPDATE WEIGHTS
        rearranged_hidden_states = np.array(self.hidden_states).reshape((len(self.hidden_states), -1)) # to package nicely into a numpy array. was a list of numpy arrays
        rearranged_del_predicts = np.flip(np.array(self.del_predicts).reshape((len(self.del_predicts), -1)), axis=0)
        rearranged_del_input_gates = np.flip(np.array(self.del_input_gates).reshape((len(self.del_input_gates), -1)), axis=0)
        rearranged_del_forget_gates = np.flip(np.array(self.del_forget_gates).reshape((len(self.del_forget_gates), -1)), axis=0)
        rearranged_del_output_gates = np.flip(np.array(self.del_output_gates).reshape((len(self.del_output_gates), -1)), axis=0)
        rearranged_del_gs = np.flip(np.array(self.del_gs).reshape((len(self.del_gs), -1)), axis=0)

        self.hidden_to_out -= np.dot(rearranged_hidden_states.T, rearranged_del_predicts)
        self.U_i -= np.dot(x.T, rearranged_del_input_gates)*lr
        self.U_f -= np.dot(x.T, rearranged_del_forget_gates)*lr
        self.U_o -= np.dot(x.T, rearranged_del_output_gates)*lr
        self.U_g -= np.dot(x.T, rearranged_del_gs)*lr

        if len(rearranged_hidden_states[:-1]):
            self.W_i -= np.dot(rearranged_hidden_states[:-1].T, rearranged_del_input_gates[:-1])*lr # rearranged_del_input_gates was flipped already, so we take everything but the last entry
            self.W_f -= np.dot(rearranged_hidden_states[:-1].T, rearranged_del_forget_gates[:-1])*lr
            self.W_o -= np.dot(rearranged_hidden_states[:-1].T, rearranged_del_output_gates[:-1])*lr
            self.W_g -= np.dot(rearranged_hidden_states[:-1].T, rearranged_del_gs[:-1])*lr

    def reset(self):
        self.input_gates = list()
        self.forget_gates = list()
        self.output_gates = list()
        self.gs = list()
        self.cell_states = list()
        self.hidden_states = list()
        self.outputs = list()

        self.del_predicts = list()
        self.del_hidden_states = list()
        self.del_cell_states = list()
        self.del_gs = list()
        self.del_input_gates = list()
        self.del_forget_gates = list()
        self.del_output_gates = list()

        self.cell_state = self.cell_state*0
        self.hidden_state = self.hidden_state*0

    # def train(self, x, y, seq_length, iterations, lr):
    #     length = len(x)
    #     for a in range(iterations):
    #         print ('iteration {0}'.format(a))
    #         for b in range(0, length, seq_length):
    #             if b == seq_length:
    #                 break;
    #             for c in range(b, seq_length):
    #                 self.forward(x[c])
    #             self.backward(x[c-seq_length+1:c+1], y[c-seq_length+1:c+1], seq_length, lr)
    #             self.reset()

    def train(self, x, y, iterations, lr):
        #BETTER WAY
        for a in range(iterations):
            print ('Iteration {}'.format(a))
            for i in range(len(x)):
                seq_length = len(x[i])
                for j in range(seq_length):
                    self.forward(x[i][j])
                self.backward(x[i], y[i], seq_length, lr)
                self.reset()


    def run(self, x):
        outputs = list()
        for row in x:
            input_gate = self.sigmoid(np.dot(row, self.U_i) + np.dot(self.hidden_state, self.W_i))
            forget_gate = self.sigmoid(np.dot(row, self.U_f) + np.dot(self.hidden_state, self.W_f))
            output_gate = self.sigmoid(np.dot(row, self.U_o) + np.dot(self.hidden_state, self.W_o))
            g = np.tanh(np.dot(row, self.U_g) + np.dot(self.hidden_state, self.W_g))
            self.cell_state = self.cell_state*forget_gate + g*input_gate
            self.hidden_state = np.tanh(self.cell_state)*output_gate
            output = self.sigmoid(np.dot(self.hidden_state, self.hidden_to_out))

            outputs.append(output)
        self.reset()
        return outputs
