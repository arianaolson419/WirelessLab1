import numpy as np 
import matplotlib.pyplot as plt

# % Open the file containing the received samples
tmp = np.fromfile("receivefile.dat", dtype=np.float32)

print("read file")
print(tmp)

# since the USRP stores the data in an interleaved fashion
# with real followed by imaginary samples 
# create a vector of half the length of the received values to store the
# data. Make every other sample the real part and the remaining samples the
# imaginary part


y = tmp[::2]+np.imag(1j)*tmp[1:2:]
mean_y = np.mean(y)
y -= mean_y

start = 910000 + 14000

end = start + 230000
print("please work: ", start, end)

# display signal
# plt.plot(np.real(y)[start:end])
# plt.show()

# Estimate the magnitude of the channel and divide the signal by this
h_mag_est = np.sqrt(np.mean(np.square(y)))
y_normalized = y / h_mag_est

# Create s[k] by squaring the signal
s = np.square(y_normalized)

# Take the FFT of the s[k] and find the frequency and amplitude of the spike.
sample_rate = 2000000
sample_period = 1/sample_rate

fft = np.fft.fft(s)

print("sample_period: \n", sample_period) # 5e-7
print("s.shape: \n", s.shape) # 11231780

freq = (np.fft.fftfreq(s.shape[-1], sample_period) * 2*np.pi)/sample_rate

plt.plot(freq, np.abs(fft))
plt.show()

# Estimate x by multiplying ~y exp(j(freq_est * k + theta_est.

# Plot constellation and determine if we need costas loop.
