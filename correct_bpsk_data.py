import numpy as np 
import matplotlib.pyplot as plt

# Open the file containing the received samples
tmp = np.fromfile("receivefile.dat", dtype=np.float32)

print("read file")
print(tmp)

# since the USRP stores the data in an interleaved fashion
# with real followed by imaginary samples 
# create a vector of half the length of the received values to store the
# data. Make every other sample the real part and the remaining samples the
# imaginary part


y = tmp[::2]+1j*tmp[1:2:]
y = y[1000:]    # Drop the first 1000 samples to account for USRP tx glitch.
mean_y = np.mean(y)
y -= mean_y

#eyeballed start & end
start = 926000 - 1000 + 120
end = start + 200000 - 1000
y = y[start:end]

# display signal
plt.plot(np.real(y))
plt.show()

# Estimate the magnitude of the channel and divide the signal by this
h_mag_est = np.sqrt(np.mean(np.square(y)))
y_normalized = y / h_mag_est
print(np.sqrt(np.mean(np.square(y_normalized))))

# Create s[k] by squaring the signal
s = np.square(y_normalized)

# Take the FFT of the s[k] and find the frequency and amplitude of the spike.
fft = np.fft.fft(s)
shifted_fft = np.fft.fftshift(fft)
freq_axis = np.linspace(-np.pi, np.pi, shifted_fft.shape[-1])

# FFT Plot
plt.plot(freq_axis, np.abs(shifted_fft))
plt.show()


print('identify offsets')
# Actual values
x_offset = freq_axis[np.argmax(shifted_fft)]
y_height = np.max(shifted_fft)
f_delta  = (x_offset)/-2                   # other peak: (.000375895)/2 
theta    = -1*np.angle(y_height)/2     # other peak: np.log(2.82879e6)/(-2 * 1j) #


# Estimate x by multiplying ~y exp(j(freq_est * k + theta_est.
psi = f_delta * np.arange(0,y_normalized.shape[-1])  + theta

# If we ignore the costas loop:
x_est = y_normalized * np.exp(1j * psi)
x_est_len = len(x_est)
# Plot constellation
plt.figure(1)
plt.plot(np.real(x_est))
plt.show(1)
plt.figure(2)
plt.plot(np.real(x_est), np.imag(x_est), '.')
plt.show(2)

prev_error = 0
x_est = []
psi_estimate = theta
beta = 0.2
alpha = 0.02

print('costas loop')
# Costas Loop
for sample in y_normalized:
    x = sample * np.exp(1j*psi_estimate)
    error = -1 * np.real(x) * np.imag(x)
    d = beta*error + alpha * (error + prev_error)
    prev_error += error
    psi_estimate = psi_estimate + d

    while psi_estimate < -np.pi:
        psi_estimate += 2*np.pi
    while psi_estimate > np.pi:
        psi_estimate -= 2*np.pi
    x_est.append(x)

x_est_len = len(x_est)

print('plotting corrected signal')
plt.subplot(2, 1, 1)
plt.plot(np.real(x_est[:x_est_len // 10]))
plt.subplot(2, 1, 2)
plt.plot(np.real(x_est)[:x_est_len // 10], np.imag(x_est)[:x_est_len // 10], '.')
plt.show()
