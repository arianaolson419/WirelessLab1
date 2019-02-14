import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal

# Elements from make_qpsk_data_file.m
N = 2000 # symbols in data, changed from 10,000
num_header_symbols = 100
symbol_period = 20
signal_len = (N + num_header_symbols) * symbol_period
original_data = np.fromfile("tx4.dat", dtype=np.float32)
original_data = original_data[::2] + 1j*original_data[1::2]
plt.suptitle("Original Data Real (above), Imaginary(below)")
plt.subplot(2, 1, 1)
plt.plot(np.real(original_data))
plt.subplot(2, 1, 2)
plt.plot(np.imag(original_data))
plt.show()

# Open the file containing the received samples.
tmp = np.fromfile("receiveQPSK_gain70.dat", dtype=np.float32)
tmp_header = np.fromfile("tx4_header_only.dat", dtype=np.float32)
y = tmp[::2]+1j*tmp[1::2]
mean_y = np.mean(y) # Correct for DC offset.
y -= mean_y
header = tmp_header[::2] + 1j*tmp_header[1:2:]

# cut off the noise 
rms_start_cutoff = .5 * np.max(y)
start_index = np.argmax(np.abs(y) > rms_start_cutoff)
end_index = int(start_index + signal_len)
if (end_index > len(y) - 1):
    end_index = len(y) - 1

cross_corr = signal.correlate(y[start_index:end_index], header)
peak_index = start_index + np.argmax(cross_corr) # index of the largest peak in y
# Make sure there is a peak from cross correlation
plt.plot(np.abs(cross_corr))
plt.show()

# take the signal from the start of the data to the known length of the data
# isolate just the data part of the signal.
print("Y data packet plot")
y = y[peak_index:peak_index + signal_len]
# plot y data packet -- real vs. imaginary
plt.suptitle("original data -- Real (above), Imaginary(below)")
plt.subplot(2, 1, 1)
plt.plot(np.real(y))
plt.subplot(2, 1, 2)
plt.plot(np.imag(y))
plt.show()


# Estimate the magnitude of the channel and divide the signal by this
h_mag_est = np.sqrt(np.mean(np.square(y)))
y_normalized = y / h_mag_est

# Create s[k] by raising the signal to the 4th power.
s = np.power(y_normalized, 4)

# Split the signal into 2 or 5 parts and calculate separate f_deltas for each one
num_chunks = 1  # Be careful to use chunks that evenly divide the data.
chunk_len = signal_len // num_chunks
for i in range(num_chunks):
    start = i * chunk_len
    fft = np.fft.fft(s[start: start + chunk_len])
    shifted_fft = np.fft.fftshift(fft)
    freq_axis = np.linspace(-np.pi, np.pi, shifted_fft.shape[-1])

    # Plot FFT to see a peak near 0
    print("FFT Plot {} / {}".format(i + 1, num_chunks))
    plt.suptitle("FFT Plot")
    plt.plot(freq_axis, np.abs(shifted_fft))
    plt.show()

    # Get f_delta and theta from finding frequency value and peak height
    x_offset = freq_axis[np.argmax(shifted_fft)]
    y_height = np.max(shifted_fft)
    f_delta  = (x_offset)/-4                   
    theta    = -1*np.angle(y_height)/4

    # Estimate x by multiplying ~y exp(j(freq_est * k + theta_est.
    psi = f_delta * np.arange(0, chunk_len)  + theta

    x_est = y_normalized[start: start + chunk_len] * np.exp(1j * psi)

# # Plot the real and imaginary waveforms
print("real vs imaginary")
plt.suptitle("received signal-- real (above) & imaginary (below)")
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(np.real(x_est))
plt.subplot(2, 1, 2)
plt.plot(np.imag(x_est))
plt.show(1)

# # Plot constellation
print("constellation plot")
plt.figure(2)
plt.suptitle("received signal-- constellation plot")
plt.plot(np.real(x_est), np.imag(x_est), '.')
plt.show(2)


# TODO: Rotate constellation by (2n +1) / 4.

# TODO: Parse the data for the bit sequence.

def turn_data_to_bits(input):
    """
    Takes in the input and samples every 20 bits. 
    If the majority of the imag or real bits there are greater than 0, assign the index a 1
    """
    bitsequence = [0] * (2*len(original_data))
    for coordinates_i in range(0, len(input), symbol_period):
        coordinates = input[coordinates_i:coordinates_i+symbol_period]
        
        # seeking for majority positive, otherwise false
        positive_real = (np.real(coordinates) > 0).sum() > (symbol_period//2) 
        positive_imag = (np.imag(coordinates) > 0).sum() > (symbol_period//2)

        if (positive_real):
            bitsequence[(coordinates_i//20)] = 1
        
        if positive_imag:
            bitsequence[(coordinates_i//20)+1] = 1

    return bitsequence

# # TODO: Compare the bit sequence with the transmitted bit sequence and get the error rate.
