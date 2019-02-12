import numpy as np 
import matplotlib.pyplot as plt

# Open the file containing the received samples.
tmp = np.fromfile("receivefile.dat", dtype=np.float32)

# since the USRP stores the data in an interleaved fashion
# with real followed by imaginary samples 
# create a vector of half the length of the received values to store the
# data. Make every other sample the real part and the remaining samples the
# imaginary part.


y = tmp[::2]+1j*tmp[1:2:]
y = y[1000:]    # Drop the first 1000 samples to account for USRP tx glitch.
mean_y = np.mean(y) # Correct for DC offset.
y -= mean_y

# TODO: cross correlate the signal with the known header to find the start of
# the data.

# TODO: take the signal from the start of the data to the known length of the
# data to isolate just the data part of the signal.

# TODO: below is the code for the eyballed signal. Delete when xcorr is done.
#eyeballed start & end
start = 926000 - 1000 + 120
end = start + 200000 - 1000
y = y[start:end]

# display signal
plt.plot(np.real(y))
plt.show()

# Estimate the magnitude of the channel and divide the signal by this.
h_mag_est = np.sqrt(np.mean(np.square(y)))
y_normalized = y / h_mag_est

# Create s[k] by raising the signal to the 4th power.
s = np.power(y_normalized, 4)

# Take the FFT of the s[k] and find the frequency and amplitude of the spike.
fft = np.fft.fft(s)
shifted_fft = np.fft.fftshift(fft)
freq_axis = np.linspace(-np.pi, np.pi, shifted_fft.shape[-1])

# FFT Plot
plt.plot(freq_axis, np.abs(shifted_fft))
plt.show()

# Actual values
x_offset = freq_axis[np.argmax(shifted_fft)]
y_height = np.max(shifted_fft)
f_delta  = (x_offset)/-4                   # other peak: (.000375895)/2 
theta    = -1*np.angle(y_height)/4     # other peak: np.log(2.82879e6)/(-2 * 1j) #

# Estimate x by multiplying ~y exp(j(freq_est * k + theta_est.
psi = f_delta * np.arange(0,y_normalized.shape[-1])  + theta

x_est = y_normalized * np.exp(1j * psi)

# Plot the real and imaginary waveforms
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(np.real(x_est))
plt.subplot(2, 1, 2)
plt.plot(np.imag(x_est))
plt.show(1)

# Plot constellation
plt.figure(2)
plt.plot(np.real(x_est), np.imag(x_est), '.')
plt.show(2)

# TODO: Parse the data for the bit sequence.

# TODO: Compare the bit sequence with the transmitted bit sequence and get
# the error rate.
