import numpy as np

from numpy.polynomial import polynomial  

def yfactor( psd_hot, psd_cold ):
    """
    y-factor is the power ratio of two power spectra
    """
    y = psd_hot / psd_cold
    return y

def temperature_system( psd_hot, psd_cold, temp_hot_kelvin, temp_cold_kelvin ):
    """
    compute the system temperature in Kelvin for two power spectra
    """
    y = yfactor( psd_hot, psd_cold )
    tsys = ( temp_hot_kelvin - y * temp_cold_kelvin ) / ( y - 1 )
    return tsys

def spectralBaseline( f_Hz, psd_db, band_edge = None, band_notches = None  ):
    """
    Determine a spectral baseline for psd in dB, returns a spectral baseline it in dB
    based on ideas of https://cds.cern.ch/record/540121/files/0203001, section 4.1
    data is assumed to be complex baseband, centered at zero Hz
    f_Hz should be monotonic, i.e. f_Hz and psd_db should have had fftshift applied
    """
    if not band_edge:
        # if no band edge is given use it all
        band_edge = np.abs(f_Hz).max()

    useidx =   np.where( np.abs(f_Hz) <= band_edge ) 

    # band notches are +/- regions to ignore
    for bn in band_notches:
        f_min, f_max = bn
        notchidx =  np.where( np.logical_and( f_Hz >= f_min, f_Hz <= f_max )) 
        useidx = np.setdiff1d( useidx, notchidx )
                             
                                          
    # fit a 2nd order polynomial to the data in db, using MHz as x-axis
    # this seems to keep the data well conditioned
    poly_fit_order = 2
    spec_fit_coeff = polynomial.polyfit( f_Hz[useidx]/1e6, psd_db[useidx] , poly_fit_order)
    spec_fit_val = polynomial.polyval( f_Hz/1e6, spec_fit_coeff )
    return spec_fit_val
