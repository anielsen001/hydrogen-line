
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

from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
import datetime
import pytz

db10 = lambda x : 10*np.log10(np.abs( x ))

doppler_vel = lambda delta_f, f_0 : speed_of_light * delta_f / f_0

# center frequency as tuned
center_freq = 1420.4e6 # MHz

#fname = '/media/apn/datadrive/hydrogen-line/mhz1420_05/limesdr_antenna_only_input.bin'
fname = '/media/apn/datadrive/hydrogen-line/mhz1420_05/limesdr_nooelec_pwroff2on_input.bin'

fnames = [ '/media/apn/datadrive/hydrogen-line/2020-05-16-firstlight/firstlight.bin',
           '/media/apn/datadrive/hydrogen-line/2020-05-16-firstlight/nadir1.bin']
fnames= glob.glob( '/media/apn/datadrive/hydrogen-line/2020-05-16-firstlight/*terminated*.bin')
#fnames= glob.glob( '/media/apn/datadrive/hydrogen-line/2020-05-16-firstlight/cool*.bin')
#fnames = glob.glob( '/media/apn/datadrive/hydrogen-line/2020-05-09-lab_tests/mhz1420_05/*bin' )
fs = 5e6 # sampling rate

#fname = '/media/apn/datadrive/hydrogen-line/mhz1420_10/limesdr_nooelec_pwron_input2.bin'
#fname = '/media/apn/datadrive/hydrogen-line/mhz1420_10/terminated_input.bin'
#fs = 10e6 #

#fname = '/media/apn/datadrive/hydrogen-line/mhz1420_20/terminated_input.bin'
#fs = 20e6 #

# offset measured in bytes
# count measured in samples
offset_sec = 0.5 # skip turn on artifacts
sec_to_read = 1
d = [ np.fromfile( fname, dtype = np.complex64, count = int(sec_to_read*fs), offset = int(8* offset_sec* fs )) for fname in fnames ]

psd =  [ welch( _d, fs = fs, nperseg =  2**10, noverlap = None, detrend = 'linear', window = 'hamming' ) for _d in d ]
pltsym = [ 'r', 'b', 'g', 'k', 'm', 'c' ]
basenames = [ os.path.basename( _f ) for _f in fnames ]

# add a second axis
# https://stackoverflow.com/questions/10514315/how-to-add-a-second-x-axis-in-matplotlib

plt.figure()
for _p, ps in zip(psd,pltsym):

    # unpack data for convenience
    f,p = _p

    # fftsshift for plotting
    f = fftshift(f)
    p = fftshift(p)

    # scale to doppler velocity
    v = doppler_vel( f, center_freq )
    
    # smooth data for plotting
    sigma = 3
    psm = ndimage.filters.gaussian_filter1d( p, 2 )
    
    #plt.plot( ( f + center_freq )/1e6, db10(  psm  ), ps )
    plt.plot( ( f  )/1e6, db10(  psm  ), ps )
    #plt.plot( v, db10(  psm  ), ps )
    
#plt.xlabel('Doppler velocity [m/s]')
plt.xlabel('Frequency [MHz]')

plt.ylabel('Uncalibrated spectrum [dB]')
plt.title('Terminated input spectra') 

plt.legend( basenames )
plt.grid()
plt.legend( ['$\mathrm{az}=128^\circ, \mathrm{elev}=45^\circ$', '$\mathrm{az}=300^\circ, \mathrm{elev}=60^\circ$']) 

# try parabola fit to:
# based on 2**10 fft size
# good band between +/- 1.6 MHz
# skip LO between +/- 0.05 MHz
useidx = np.where( np.logical_and( np.abs(f)<1.6e6 , np.abs(f)>0.05e6) )[0]
poly_fit_order=2
# tried 4 and results didn't work well. 
psm_fit = polynomial.polyfit( f[useidx]/1e6, db10(psm[useidx]), poly_fit_order )
psm_fit_val = polynomial.polyval( f/1e6, psm_fit)   
#useidx = np.where(  np.abs(f)>0.05 )[0]
plt.plot( f[useidx]/1e6, db10(psm[useidx]), 'x')
plt.plot( f/1e6, psm_fit_val, 'r')

# call the zero gain of f/psm_fit at center_freg, which happens to be the
# zero freq. We can find the index of zero frequecy with
center_freq_idx = np.argmin( np.abs( f ) )
plt.figure(); plt.plot( f/1e6, psm_fit_val - psm_fit_val[center_freq_idx])

plt.figure();
plt.plot( f/1e6, db10(psm) ) 
plt.plot( f/1e6, psm_fit_val)
plt.plot( f/1e6, db10(psm) - (psm_fit_val - psm_fit_val[center_freq_idx]) ) 
plt.grid()
plt.legend( [ 'original', 'correction fit', 'correction removed'] )

f, t, Sxx= spectrogram(d, fs = fs, nperseg =  2**16, noverlap = 2**15)

plt.figure();
plt.imshow( fftshift( db10( Sxx ), axes = 0 ),
            aspect = 'auto',
            cmap = 'cubehelix',
            extent = [ t.min(), t.max(), f.min(), f.max() ],
            vmin = -140, vmax = -90 )  
plt.colorbar()


dayton = EarthLocation( lat = 39.718250, lon = -84.164360, height = 260 *u.m )

# https://stackoverflow.com/questions/17492923/reading-data-from-csv-into-pandas-when-date-and-time-are-in-separate-columns
obs = pd.read_csv('/media/apn/datadrive/hydrogen-line/2020-05-16-firstlight/2020-05-16-firstlight - obs.csv', parse_dates={'datetime':['date_local','time_local']})

# convert from local to utc
obs['datetime'] = pd.DatetimeIndex( obs['datetime'] ).tz_localize('US/Eastern').tz_convert('UTC') 

# go row by row to find ra/dec for each az/alt pair
def altaz2radec( row ):
    # takes one row of obs and computes both ra/dec
    loc = EarthLocation( lat = row['latitude'] , lon = row['longitude'], height = row['height']*u.m )
    obs_time = Time( row['datetime'] )
    #aa = AltAz( location = loc, obstime = obs_time )
    sc = SkyCoord( alt = row['altitude']*u.deg, az = row['azimuth']*u.deg, location = loc, obstime = obs_time , frame = 'altaz' )

    row['ra'] = sc.icrs.ra.deg
    row['dec'] = sc.icrs.dec.deg

    return row

# https://stackoverflow.com/questions/23586510/return-multiple-columns-from-pandas-apply
obs.apply( altaz2radec, axis = 1 )
    
loc = EarthLocation( lat = obs['latitude'], lon = obs['longitude'], height = obs['height']*u.m )
