from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

def get_GPS_data(filepath, year, time, name):

    # Load data from GPS station matlab file
    matlab_data = loadmat(filepath + 'GPS' + str(year) + '_' + str(time) + 'h_' + str(name) + '.mat')
    
    # Get the receiver name
    receiver_name = matlab_data['stn_receiver_' + str(time) + 'h'][0][0][0]

    # Get data from each of the struct entries within the Matlab file
    t_array = matlab_data['stn'+ str(receiver_name) + '_' + str(time) + 'h'][0][0][1]
    x_array = matlab_data['stn'+ str(receiver_name) + '_' + str(time) + 'h'][0][0][2]
    y_array = matlab_data['stn'+ str(receiver_name) + '_' + str(time) + 'h'][0][0][3]
    z_array = matlab_data['stn'+ str(receiver_name) + '_' + str(time) + 'h'][0][0][4]
    xerr_array = matlab_data['stn'+ str(receiver_name) + '_' + str(time) + 'h'][0][0][5]
    yerr_array = matlab_data['stn'+ str(receiver_name) + '_' + str(time) + 'h'][0][0][6]
    zerr_array = matlab_data['stn'+ str(receiver_name) + '_' + str(time) + 'h'][0][0][7]
    v_array = matlab_data['stn'+ str(receiver_name) + '_' + str(time) + 'h'][0][0][8]
    verr_array = matlab_data['stn'+ str(receiver_name) + '_' + str(time) + 'h'][0][0][9]

    # Flatten the data array - the arrays captured earlier look like [[x1], [x2], [x3], ...]
    # and we want them like [x1, x2, x3, ...] so we flatten them to do so
    t = t_array.flatten()
    x = x_array.flatten()
    y = y_array.flatten()
    z = z_array.flatten()
    v = v_array.flatten()
    xerr = xerr_array.flatten()
    yerr = yerr_array.flatten()
    zerr = zerr_array.flatten()
    verr = verr_array.flatten()

    gps_data = np.array([t, x, y, z, v, xerr, yerr, zerr, verr])
    return gps_data


#GPS2011_1h_GULL = get_GPS_data(2011, 1, 'GULL')
#GPS2012_1h_GULL = get_GPS_data(2012, 1, 'GULL')
#
#t2011 = GPS2011_1h_GULL[0]
#v2011 = GPS2011_1h_GULL[4]
#t2012 = GPS2012_1h_GULL[0]
#v2012 = GPS2012_1h_GULL[4]
#
#meanv_2011 = np.nanmean(v2011)
#meanv_2012 = np.nanmean(v2012)
#
#fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (20,6))
#
#ax.plot(t2011, v2011, 'b+', label = '2011')
#ax.plot(t2012, v2012, 'g+', label = '2012')
#ax.axhline(y = meanv_2011, color = 'red', linewidth = 1, label = 'mean 2011 vel')
#ax.axhline(y = meanv_2012, color = 'orange', linewidth = 1, label = 'mean 2012 vel')
#
#ax.set_xlabel('Day')
#ax.set_ylabel('Velocity (m/yr)')
#ax.set_title('Velocity profile at GULL borehole')
#ax.legend()
#
#plt.show()


