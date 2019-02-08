% Open the file containing the received samples
f2 = fopen('tx2.dat', 'rb');

% read data from the file
tmp = fread(f2, 'float32');

% close the file
fclose(f2);


% since the USRP stores the data in an interleaved fashion
% with real followed by imaginary samples 
% create a vector of half the length of the received values to store the
% data. Make every other sample the real part and the remaining samples the
% imaginary part
y = zeros(length(tmp)/2,1);
y = tmp(1:2:end)+j*tmp(2:2:end);

% Identify data packet
% Estimate the magnitude of the channel and divide the signal by this
% Create s[k] by squaring the signal
% Take the FFT of the s[k] and find the frequency and amplitude of the spike.
% Estimate x by multiplying ~y exp(j(freq_est * k + theta_est.
% Plot constellation and determine if we need costas loop.
