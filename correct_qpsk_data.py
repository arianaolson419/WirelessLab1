import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal

# Elements from make_qpsk_data_file.m
N = 1000 # symbols in data, changed from 10,000
num_header_symbols = 100
symbol_period = 20
signal_len = (N + num_header_symbols) * symbol_period
data_length = N * symbol_period
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

tmp = np.fromfile("tx4_data_only.dat", dtype=np.float32)
data_only = tmp[::2] + 1j*tmp[1::2]

# cut off the noise 
def cut_noise(input_signal):
    start_cutoff = .5 * np.max(y)
    rough_start_index = np.argmax(np.abs(y) > start_cutoff)
    rough_end_index = int(rough_start_index + signal_len)
    if (rough_end_index > len(y) - 1):
        rough_end_index = len(y) - 1

    cross_corr = signal.correlate(input_signal[rough_start_index:rough_end_index], header)
    #cross_corr = signal.correlate(input_signal[rough_start_index:rough_end_index], header, 'same')
    peak_index = rough_start_index + np.argmax(cross_corr)# index of the largest peak in y # Make sure ther is a peak from cross correlation
    #peak_index = rough_start_index + np.argmax(cross_corr)  - (num_header_symbols * symbol_period // 2) - 1 # index of the largest peak in y # Make sure ther is a peak from cross correlation
    plt.plot(np.abs(cross_corr))
    plt.show()
    y_tmp = np.copy(input_signal)
    y_tmp[:peak_index] = 0
    y_tmp[peak_index + signal_len:] = 0
    plt.plot(input_signal.real)
    plt.plot(y_tmp.real)
    plt.title('Compare signal')
    plt.show()
    # This seems to cut off the header.
    input_signal = input_signal[peak_index:peak_index + data_length]
    #input_signal = input_signal[rough_start_index:rough_start_index + signal_len]
    return input_signal

y = cut_noise(y)
# plot y data packet -- real vs. imaginary
plt.suptitle("original data -- Real (above), Imaginary(below)")
plt.subplot(2, 1, 1)
plt.plot(np.real(y))
plt.subplot(2, 1, 2)
plt.plot(np.imag(y))
plt.show()

# TODO: Split the signal into 2 or 5 parts and calculate separate f_deltas for each one

num_parts = 1
x_est = np.zeros(y.shape[-1], dtype=np.complex64)
segmentLength = (len(y)//num_parts)

for i in range(num_parts):
    startIndex = i * segmentLength
    endIndex = startIndex + segmentLength
    y_segment = y[startIndex:endIndex]
    h_mag_est = np.sqrt(np.mean(np.square(y_segment)))
    y_normalized = y_segment / h_mag_est

    # Create s[k] by raising the signal to the 4th power.
    plt.subplot(2, 1, 1)
    plt.plot(y_normalized.real)
    plt.title('Before FFT')
    plt.subplot(2, 1, 2)
    plt.plot(y_normalized.imag)
    plt.show()
    s = np.power(y_normalized, 4)
    fft = np.fft.fft(s)
    shifted_fft = np.fft.fftshift(fft)
    freq_axis = np.linspace(-np.pi, np.pi, shifted_fft.shape[-1])

    # Plot FFT to see a peak near 0
    print("FFT Plot")
    plt.suptitle("FFT Plot")
    plt.plot(freq_axis, np.abs(shifted_fft))
    plt.show()

    # Get f_delta and theta from finding frequency value and peak height
    x_offset = freq_axis[np.argmax(shifted_fft)]
    y_height = np.max(shifted_fft)
    y_scale = y_height / np.abs(y_height)
    print('y height:', y_height)
    f_delta  = (x_offset)/-4                   
    theta    = -1*np.angle(y_height)/4

    # # Estimate x by multiplying ~y exp(j(freq_est * k + theta_est.
    psi = f_delta * np.arange(0,y_normalized.shape[-1])
    x_est[startIndex:endIndex] = y_normalized * np.exp(1j * psi) / np.power(y_scale, 0.25)

# # Plot the real and imaginary waveforms
#print("real vs imaginary")
#plt.suptitle("received signal-- real (above) & imaginary (below)")
#plt.figure(1)
#plt.subplot(2, 1, 1)
#plt.plot(np.real(x_est))
#plt.subplot(2, 1, 2)
#plt.plot(np.imag(x_est))
#plt.show(1)

def rotate_constellation_quadrants(angle, signal):
    """Rotates the constellation by the given angle, in radians, in order to
    shift the quadrants of the data.
    """
    return signal * np.exp(1j * angle)

def turn_data_to_bits(input):
    """
    Takes in the input and samples every 20 bits. 
    If the majority of the imag or real bits there are greater than 0, assign the index a 1
    """
    bitsequence = [0] * (2 * input.shape[-1] // symbol_period)
    #bitsequence = [0] * (2*len(original_data))
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

# Rotate the signal
x_est = rotate_constellation_quadrants(5 * np.pi / 4,  x_est)
plt.plot(x_est.real)
plt.show()

# # Plot constellation
print("constellation plot")
plt.figure(2)
plt.suptitle("received signal-- constellation plot")
plt.plot(np.real(x_est), np.imag(x_est), '.')
plt.show(2)

originaldata_bits = turn_data_to_bits(data_only[:N * symbol_period])
receiveddata_bits = turn_data_to_bits(x_est)
wrongbits = []
print(len(originaldata_bits), len(receiveddata_bits))
for i in range(len(originaldata_bits)):
    if originaldata_bits[i] != receiveddata_bits[i]:
        wrongbits.append(i)

#print("wrongbits: ", wrongbits)
print("original data len: ", len(originaldata_bits))
print(len(wrongbits)/len(originaldata_bits))

plt.plot(wrongbits, ".")
plt.show()

# TODO: Figure out why approximately half of the first 15000 bits are wrong
