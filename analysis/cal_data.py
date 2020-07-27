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

fnames_cal = glob.glob( '/media/apn/datadrive/hydrogen-line/2020-05-16-firstlight/*terminated*.bin')

fs = 5e6 # sampling rate

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

cal_fit_db = spectralBaseline( f, db10(p), band_edge = 1.6e6, band_notches = [[ -0.05e6, 0.05e6 ]] )

#useidx = np.where( np.logical_and( np.abs(f)<1.6e6 , np.abs(f)>0.05e6) )[0]
#poly_fit_order=2
# tried 4 and results didn't work well. 
#cal_fit = polynomial.polyfit( f[useidx]/1e6, db10(p[useidx]), poly_fit_order )
#cal_fit_val = polynomial.polyval( f/1e6, cal_fit)


center_freq_idx = np.argmin( np.abs( f ))
cal = cal_fit_db - cal_fit_db[ center_freq_idx ] 


plt.plot( f/1e6, db10(p), 'x')
plt.plot( f/1e6, cal_fit_db, 'r')


# read the measurement data
fnames_meas = glob.glob( '/media/apn/datadrive/hydrogen-line/2020-05-16-firstlight/cool*.bin')
offset_sec = 0.5 # skip turn on artifacts
sec_to_read = 1
d = [ np.fromfile( fname, dtype = np.complex64, count = int(sec_to_read*fs), offset = int(8* offset_sec* fs )) for fname in fnames_meas ]

psd =  [ welch( _d, fs = fs, nperseg =  2**10, noverlap = None, detrend = 'linear', window = 'hamming', return_onesided = False ) for _d in d ]
pltsym = [ 'r', 'b', 'g', 'k', 'm', 'c' ]
basenames = [ os.path.basename( _f ) for _f in fnames_meas ]

# apply the cal to each psd and plot
fig, ax = plt.subplots()
for _psd in psd:
    _f, _p = _psd
    _f = fftshift( _f )
    _p = db10( fftshift( _p ) )

    #_p_cal = _p - cal
    ax.plot( _f/1e6 , _p-cal )

ax.grid()
ax.set_xlabel( 'Frequency [MHz]')
ax.set_title('Filter corrected cold spectra')

ax.legend( ['$\mathrm{az}=128^\circ, \mathrm{elev}=45^\circ$',
            '$\mathrm{az}=300^\circ, \mathrm{elev}=60^\circ$']) 
#v = doppler_vel( f, center_freq )

dopp = lambda f: doppler_vel( _f, center_freq )
ax2 = ax.secondary_xaxis('top' , functions = (dopp,dopp) )

ax2.set_xlabel('Doppler velocity [km/s]')
