import numpy as np 
# import matplotlib
# import matplotlib.pyplot as plt


N = 10000
# we need 100 random bits of values +- 1
bits = np.sign(np.random.randn(N))
symbol_period = 20

# create a generic pulse of unit height 
# with width equal to symbol period
pulse = np.ones(symbol_period) #symbol_period number of ones

# spread out the values in "bits" by symbol_period
# first create a vector to store the values
x = np.zeros(symbol_period*len(bits))

# assign every symbol_period-th sample to equal a value from bits
x[::symbol_period] = bits

# now convolve the single generic pulse with the spread-out bits
x_tx = np.convolve(pulse, x)

# # to visualize, make a stem plot (not working for some reason)
# plt.stem(x_tx)
# plt.show()

# zero pad the beginning with 100000 samples to ensure that any glitch that
# happens when we start transmitting doesn't effect the data
x_tx = np.pad(x_tx, (100000, 100000),'constant')

# writing the data into a format that USRP can understand
# specifically using float32 numbers with real values followed by imag ones

# step 1: creating a vector to store interleaved real & imag values
tmp = np.zeros(len(x_tx)*2)

# step 2: assign the real part of x_tx to every other sample 
# and the imag part to the remaining example
tmp[::2] = np.real(x_tx)
tmp[1::2] = np.imag(x_tx)

# step 3: open a file to write in binary format -- having trouble with this!
f1 = open("t2.dat", "wb")
# write the values as a float32
f1.write(tmp/2)
# close the file
f1.close()