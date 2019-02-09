import numpy as np 
import matplotlib.pyplot as plt

# % Open the file containing the received samples
tmp = np.fromfile("receivefile.dat")

print("read file")
print(tmp)

# % since the USRP stores the data in an interleaved fashion
# % with real followed by imaginary samples 
# % create a vector of half the length of the received values to store the
# % data. Make every other sample the real part and the remaining samples the
# % imaginary part


y = tmp[::2]+np.imag(1j)*tmp[1:2:]

# plt.plot(np.real(y))
# plt.show()

start = 0
end = len(y)
cutoff = (10**(-19))
# # % Identify data packet
# indexing reference: start:end:step
for i in range(len(y))[0:len(y):20]:
    if (np.average(y[i:i+20]) > cutoff):
        start = i
        break

for i in range(start, len(y))[start:len(y):20]:
    if (np.average(y[i:i+20]) < cutoff):
        end = i + 20
        break
print("please work: ", start, end)

plt.plot(np.real(y)[start:end])
plt.show()

plt.plot(np.real(y))
plt.show()

# # % Estimate the magnitude of the channel and divide the signal by this
# # % Create s[k] by squaring the signal
# # % Take the FFT of the s[k] and find the frequency and amplitude of the spike.
# # % Estimate x by multiplying ~y exp(j(freq_est * k + theta_est.
# # % Plot constellation and determine if we need costas loop.
