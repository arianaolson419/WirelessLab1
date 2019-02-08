% Open the file containing the received samples
f2 = fopen('tx4.dat', 'rb');

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
%stem(real(y(end-1000:end)));
plot(real(y), imag(y), 'o');

% to visualize, plot the real and imaginary parts separately
%return;
%subplot(211)
%stem(real(y));
%subplot(212)
%stem(imag(y));
%plot(real(y(end - 10000:end)), imag(y(end - 10000:end)), 'o');
