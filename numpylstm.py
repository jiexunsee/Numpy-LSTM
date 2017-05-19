import numpy as np


''' An LSTM '''
class LSTM:
    def sigmoid(self, x):
        return (1/(1+np.exp(-x)))

    def sigmoid_derivative(self, x):
        return (x*(1-x))

    def tanh_derivative(self, x):
        return (1-(np.tanh(x))**2)

    def __init__(self, input_dim, output_dim):

        self.U_i = np.random.uniform(-0.1, 0.1, size=(input_dim, output_dim))
        self.U_f = np.random.uniform(-0.1, 0.1, size=(input_dim, output_dim))
        self.U_o = np.random.uniform(-0.1, 0.1, size=(input_dim, output_dim))
        self.U_g = np.random.uniform(-0.1, 0.1, size=(input_dim, output_dim))
        self.W_i = np.random.uniform(-0.1, 0.1, size=(output_dim, output_dim))
        self.W_f = np.random.uniform(-0.1, 0.1, size=(output_dim, output_dim))
        self.W_o = np.random.uniform(-0.1, 0.1, size=(output_dim, output_dim))
        self.W_g = np.random.uniform(-0.1, 0.1, size=(output_dim, output_dim))

        self.cell_state = np.zeros((1, output_dim))
        self.output_state = np.zeros((1, output_dim))

        # caching for backpropagation
        self.input_gates = list()
        self.forget_gates = list()
        self.output_gates = list()
        self.gs = list()
        self.cell_states = list()
        self.output_states = list()

        # caching for backpropagation
        self.del_output_states = list()
        self.del_cell_states = list()
        self.del_gs = list()
        self.del_input_gates = list()
        self.del_forget_gates = list()
        self.del_output_gates = list()

    def forward(self, x):
        input_gate = self.sigmoid((x @ self.U_i)+(self.output_state @ self.W_i)) # a short form for np.dot()
        # input_gate = self.sigmoid(np.dot(x, self.U_i) + np.dot(self.output_state, self.W_i))
        forget_gate = self.sigmoid(np.dot(x, self.U_f) + np.dot(self.output_state, self.W_f))
        output_gate = self.sigmoid(np.dot(x, self.U_o) + np.dot(self.output_state, self.W_o))
        g = np.tanh(np.dot(x, self.U_g) + np.dot(self.output_state, self.W_g))
        self.cell_state = self.cell_state*forget_gate + g*input_gate
        self.output_state = np.tanh(self.cell_state)*output_gate

        self.input_gates.append(input_gate)
        self.forget_gates.append(forget_gate)
        self.output_gates.append(output_gate)
        self.gs.append(g)
        self.cell_states.append(self.cell_state)
        self.output_states.append(self.output_state)

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
            error = self.output_states[seq_length-a-1] - y[seq_length-a-1]
            del_output_state = error + timestep_error
            del_cell_state = del_output_state*self.output_gates[seq_length-a-1]*self.tanh_derivative(self.cell_states[seq_length-a-1]) + future_del_cell_state*future_forget_gate
            del_g = del_cell_state*self.input_gates[seq_length-a-1]*self.tanh_derivative(self.gs[seq_length-a-1])
            # del_g = del_cell_state*self.input_gates[seq_length-a-1]*(1-(self.gs[seq_length-a-1])**2)
            del_input_gate = del_cell_state*self.gs[seq_length-a-1]*self.sigmoid_derivative(self.input_gates[seq_length-a-1])
            del_forget_gate = del_cell_state*prev_cell_state*self.sigmoid_derivative(self.forget_gates[seq_length-a-1])
            del_output_gate = del_output_state*np.tanh(self.cell_states[seq_length-a-1])*self.sigmoid_derivative(self.output_gates[seq_length-a-1])

            timestep_error = (np.dot(self.W_i, del_input_gate.T) + np.dot(self.W_f, del_forget_gate.T) + np.dot(self.W_o, del_output_gate.T) + np.dot(self.W_g, del_g.T)).T
            future_del_cell_state = del_cell_state

            self.del_output_states.append(del_output_gate)
            self.del_cell_states.append(del_cell_state)
            self.del_gs.append(del_g)
            self.del_input_gates.append(del_input_gate)
            self.del_forget_gates.append(del_forget_gate)
            self.del_output_gates.append(del_output_gate)

        #UPDATE WEIGHTS
        self.U_i -= np.dot(x.T, np.flip(np.array(self.del_input_gates).reshape((len(self.del_input_gates), -1)), axis=0))*lr
        self.U_f -= np.dot(x.T, np.flip(np.array(self.del_forget_gates).reshape((len(self.del_forget_gates), -1)), axis=0))*lr
        self.U_o -= np.dot(x.T, np.flip(np.array(self.del_output_gates).reshape((len(self.del_output_gates), -1)), axis=0))*lr
        self.U_g -= np.dot(x.T, np.flip(np.array(self.del_gs).reshape((len(self.del_gs), -1)), axis=0))*lr

        if self.output_states[:-1]:
            reshaped_output_states = np.array(self.output_states[:-1]).reshape((len(self.output_states[:-1]),-1)).T
            self.W_i -= np.dot(reshaped_output_states, np.flip(np.atleast_2d(np.squeeze(self.del_input_gates[1:])), axis=0))*lr
            self.W_f -= np.dot(reshaped_output_states, np.flip(np.atleast_2d(np.squeeze(self.del_forget_gates[1:])), axis=0))*lr
            self.W_o -= np.dot(reshaped_output_states, np.flip(np.atleast_2d(np.squeeze(self.del_output_gates[1:])), axis=0))*lr
            self.W_g -= np.dot(reshaped_output_states, np.flip(np.atleast_2d(np.squeeze(self.del_output_gates[1:])), axis=0))*lr

    def reset(self):
        self.input_gates = list()
        self.forget_gates = list()
        self.output_gates = list()
        self.gs = list()
        self.cell_states = list()
        self.output_states = list()

        self.del_output_states = list()
        self.del_cell_states = list()
        self.del_gs = list()
        self.del_input_gates = list()
        self.del_forget_gates = list()
        self.del_output_gates = list()

        self.cell_state = self.cell_state*0
        self.output_state = self.output_state*0

    def train(self, x, y, seq_length, iterations, lr):
        length = len(x)
        for a in range(iterations):
            print ('iteration {0}'.format(a))
            for b in range(0, length, seq_length):
                if b == seq_length:
                    break;
                for c in range(b, seq_length):
                    self.forward(x[c])
                self.backward(x[c-seq_length+1:c+1], y[c-seq_length+1:c+1], seq_length, lr)
                self.reset()

        #BETTER WAY
        # for a in range(iterations):
        #     for (x, y) in data:
        #         seq_length = len(x)
        #         for i in seq_length:
        #             self.forward(x[i])
        #         self.backward(x, y, seq_length, lr)
        #         self.reset_lists()



    def run(self, x):
        output = list()
        for row in x:
            input_gate = self.sigmoid(np.dot(row, self.U_i) + np.dot(self.output_state, self.W_i))
            forget_gate = self.sigmoid(np.dot(row, self.U_f) + np.dot(self.output_state, self.W_f))
            output_gate = self.sigmoid(np.dot(row, self.U_o) + np.dot(self.output_state, self.W_o))
            g = np.tanh(np.dot(row, self.U_g) + np.dot(self.output_state, self.W_g))
            self.cell_state = self.cell_state*forget_gate + g*input_gate
            self.output_state = np.tanh(self.cell_state)*output_gate
            output.append(self.output_state)
        self.reset()
        return output

if __name__ == '__main__':
    # lstm = LSTM(2, 1)
    lstm = LSTM(2, 2)

    # lstm.train(np.array([[1,2], [1,0]]), [[1],[0]], 2, 5000, 0.05)
    lstm.train(np.array([[1,2], [1,0]]), [[-1,-1],[1,1]], 2, 5000, 0.01)
    output = lstm.run(np.array([[1,2], [1,0]]))
    output2 = lstm.run(np.array([[1,2], [1,0]]))
    
    print (output)
    print (output2)

    # int2binary = {}
    # binary_dim = 8
    #
    # largest_number = pow(2,binary_dim)
    # binary = np.unpackbits(
    #     np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
    # for i in range(largest_number):
    #     int2binary[i] = binary[i]
