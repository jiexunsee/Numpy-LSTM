import numpy as np
from numpylstm import LSTM

def lame_test():
    lstm = LSTM(2, 5, 1)

    lstm.train([np.array([[1,2], [1,0]])], [[[0.3],[0.6]]], 3000, 0.5)
    output = lstm.run(np.array([[1,2], [1,0]])) # should give close to [[[0.3],[0.6]]]

    print (output) # not even a toy example, but seems to be working

''' TESTING USING BINARY ADDITION. Gives runtime warnings of 'overflow encountered in exp' and 'invalid value encountered in multiply'. Eventual prediction is nan  '''
def binary_test():
    int2binary = {}
    binary_dim = 8

    largest_number = pow(2,binary_dim)
    binary = np.unpackbits(
        np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
    for i in range(largest_number):
        int2binary[i] = binary[i]

    # generate training data of binary additions
    nums = list()
    total = list()
    for j in range(10):
        a_int = np.random.randint(largest_number/2) # int version
        a = np.atleast_2d(int2binary[a_int]) # binary encoding
        b_int = np.random.randint(largest_number/2) # int version
        b = np.atleast_2d(int2binary[b_int]) # binary encoding
        c_int = a_int + b_int
        c = int2binary[c_int]

        x = np.concatenate((a, b), axis=0)
        nums.append(x.T)
        total.append(np.split(c, binary_dim))

    lstm = LSTM(2, 10, 1)
    lstm.train(nums, total, 300, 0.05)

    # TESTING
    a_int = np.random.randint(largest_number/2) # int version
    a = np.atleast_2d(int2binary[a_int]) # binary encoding
    b_int = np.random.randint(largest_number/2) # int version
    b = np.atleast_2d(int2binary[b_int]) # binary encoding
    c_int = a_int + b_int
    c = int2binary[c_int]
    x = np.concatenate((a, b), axis=0)
    result = np.squeeze(lstm.run(x.T))
    print (result) # prediction
    print (c) # correct result

if __name__ == '__main__':
    lame_test()
