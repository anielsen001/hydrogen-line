import numpy as np
from scipy.signal import spectrogram, welch
from scipy.fft import fftshift
from scipy import interpolate
from scipy import ndimage
from scipy.constants import speed_of_light
import matplotlib.pylab as plt; plt.ion()
import os
import glob
import pandas as pd
from numpy.polynomial import polynomial

from spectralbaseline import spectralBaseline

from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
import datetime
import pytz

db10 = lambda x : 10*np.log10(np.abs( x ))

doppler_vel = lambda delta_f, f_0 : speed_of_light * delta_f / f_0

# center frequency as tuned
center_freq = 1420.4e6 # MHz

fnames_cold_300 = glob.glob( '/media/apn/datadrive/hydrogen-line/2020-05-16-firstlight/cool2.bin')
fnames_cold_140 = glob.glob( '/media/apn/datadrive/hydrogen-line/2020-05-16-firstlight/cool1.bin')
fnames_hot = glob.glob( '/media/apn/datadrive/hydrogen-line/2020-05-16-firstlight/nadir1.bin' )
t_cold_kelvin = 10.0
t_hot_kelvin = 300.0

fs = 5e6 # sampling rate

# read file and get psd
def file2psd( fname, fs, offset_sec = 0.0, sec_to_read = -1,
              nperseg = 2**10, detrend = 'linear', window = 'hamming' ):
    data = np.fromfile( fname, dtype = np.complex64, count = int(sec_to_read*fs), offset = int(8* offset_sec* fs ))

    f,p = welch( data, fs = fs, nperseg = nperseg, noverlap = None, detrend = detrend, window = window, return_onesided = False )
    
    f = fftshift(f)
    p = fftshift(p)
    
    return f,p 

f,p_hot = file2psd( fnames_hot[0], fs, sec_to_read = 1.0 )
f,p_cold_300 = file2psd( fnames_cold_300[0], fs, sec_to_read = 1.0 )
f,p_cold_140 = file2psd( fnames_cold_140[0], fs, sec_to_read = 1.0 )

tsys_300 = temperature_system( p_hot, p_cold_300, 300, 10 )
tsys_140 = temperature_system( p_hot, p_cold_140, 300, 10 )

# offset measured in bytes
# count measured in samples
offset_sec = 0.5 # skip turn on artifacts
sec_to_read = 1
d = [ np.fromfile( fname, dtype = np.complex64, count = int(sec_to_read*fs), offset = int(8* offset_sec* fs )) for fname in fnames_cal ]

psd =  [ welch( _d, fs = fs, nperseg =  2**10, noverlap = None, detrend = 'linear', window = 'hamming', return_onesided = False ) for _d in d ]
# psd is a list each element is f,p
# fftsshift for ease of use
f,p = psd[0]
f = fftshift(f)
p = fftshift(p)



cool_fit_db = spectralBaseline( f, db10(p), band_edge = 1.6e6, band_notches = [[ -0.05e6, 0.05e6 ], [-0.2e6, 0.4e6] ] )

plt.figure()
plt.plot( f/1e6, db10(p) )
plt.plot( f/1e6, cool_fit_db )

zero_freq_idx = np.argmin( np.abs( f ) )
plt.figure()
plt.plot( f/1e6, db10(p) - cal_fit_db + cal_fit_db[zero_freq_idx] )
plt.plot( f/1e6, cool_fit_db- cal_fit_db + cal_fit_db[zero_freq_idx])

cool_fit_db_cal = spectralBaseline( f, cool_fit_db- cal_fit_db + cal_fit_db[zero_freq_idx] , band_edge = 1.6e6, band_notches = [[ -0.05e6, 0.05e6 ], [-0.2e6, 0.4e6] ] )

plt.figure(6)
plt.plot( f/1e6, cool_fit_db_cal )
