import os
import tomllib
import datetime
import logging
from collections import namedtuple
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import xarray as xr
import numpy as np
import metpy
from metpy.units import units

PATH_TO_LOGCFG = "config/logging.conf"

with open(os.path.join(os.getcwd(), PATH_TO_LOGCFG), 'rb') as log_cfg:
	logging.config.dictConfig(tomllib.load(log_cfg))

global log
log = logging.getLogger("main.calc")

def calculate_min_contrail_heights(data_file):

	now = pd.Timestamp.now(tz='UTC')
	log.info(f"Starting file parsing routine at {now} UTC")

	try:
		data = xr.load_dataset(data_file, 
							   engine='cfgrib', 
							   backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa'},'errors':'ignore'})						 
	except:
		log.fatal("Could not open dataset")
		raise RuntimeError("Could not open dataset")
	else:
		log.info("Successfully opened data file")

	#print(data)

	latitude = data.latitude.data
	longitude = data.longitude.data
	np_dt_obj = data.valid_time.data
	init_time = pd.to_datetime(data.time.data)
	dt_obj = pd.to_datetime(np_dt_obj)

	fcst_hr = round((dt_obj - init_time).total_seconds() / 3600)
	dt_str = str(dt_obj)
	
	log.info(f"Parsing time {dt_str}")
	
	# Pressure, temp, height, RH, specific humidity data
	press = data.isobaricInhPa.data
	tmp = data.t.data-273.15 #Kelvin to degC
	hgt = data.gh.data*3.28084 # GPM meters to feet
	rh = data.r.data
	# Calc specific humidity from RH
	tmp2 = tmp * units.degC
	rh2 = rh * units.percent
	dpt = metpy.calc.dewpoint_from_relative_humidity(tmp2, rh2) #dewpoint (degC)
	
	# Make 3d pressure array
	#Creating a dummy array of same size
	arr = np.zeros(tmp.shape)
	z = [] 
	for element in range(len(press)):
		z.append(arr[element,:,:] + press[element])			
	#Stack the list of arrays into master array
	arr_final = np.dstack(z)
	#Rearranges so that the dimensions match y
	arr_final = np.rollaxis(arr_final,-1)
	press3d = arr_final 
	press3d_units = press3d * units.hPa
	
	# specific humidity (g/kg)
	q = metpy.calc.specific_humidity_from_dewpoint(press3d_units, dpt).to('g/kg') 
	q = q.magnitude #removes units
	
	x = -93.9 + (4.92*np.log(press)) + (0.45*np.log(press))**2
	y = (0.30*rh) - (0.0074*rh)**2 + (0.000053*rh)**3 # John's Formula Using RH
	
	###### Elizabeth's Formula ######
	#y = 0.30*(100*((q/(1-q))/(.62197*((6.1094*np.exp((17.625*(tmp-273))/(tmp-29.96)))/((press_new - 6.1094)*np.exp((17.625*(tmp-273))/(tmp-29.96))))))) - 0.0074*(100*((q/(1-q))/(.62197*((6.1094*np.exp((17.625*(tmpc-273))/(tmp-29.96)))/((press_new - 6.1094)*np.exp((17.625*(tmp-273))/(tmp-29.96))))))**2) + 0.000053*(100*((q/(1-q))/(.62197*((6.1094*np.exp((17.625*(tmp-273))/(tmp-29.96)))/((press_new - 6.1094)*np.exp((17.625*(tmp-273))/(tmp-29.96))))))**3)
	# satvaporpress = 6.1094*np.exp((17.625*(tmp-273))/(tmp-29.96))
	# a = press_new - satvaporpress
	# b = satvaporpress / a
	# c = b * .62197
	# w = (q) / (1 - q)
	# rh_eliz = (w/c) * 100
	# y = (0.30*rh_eliz) - (0.0074*rh_eliz)**2 + (0.000053*rh_eliz)**3 # John's Formula Using Elizabeth's RH
	
	### For Testing Purposes ###
	# tmpk = 243 #Kelvin
	# q = 0.0003 #kg/kg
	# press = 600 #mb
	# satvaporpress = 6.1094*np.exp((17.625*(tmpk-273))/(tmpk-29.96))
	# a = press - (satvaporpress)
	# b = (satvaporpress) / a
	# c = b * .62197
	# w = ((q) / (1 - q))
	# rh_eliz = (w/c) * 100
	# satvaporpress = np.exp((17.625*(tmpk-273))/(tmpk-29.96))
	# rh = (.263*(press*100)*w)/(satvaporpress)
	# # q = ((12.674) / (1000 - 12.674)) * 621.97
	# w = q/(1-q)
	# diff = y_old - y
	
	#################################
 
	# Make 3d array of x parameter
	#Creating a dummy array of same size
	arr = np.zeros(tmp.shape)
	z = [] 
	for element in range(len(press)):
		z.append(arr[element,:,:] + x[element])			
	#Stack the list of arrays into master array
	arr_final = np.dstack(z)
	#Rearranges so that the dimensions match y
	arr_final = np.rollaxis(arr_final,-1)
	x_new = arr_final 
	
	#Critical temperature for contrail formation
	t_crit = x_new + y
	
	# Contrail Potential based on Temperature Difference
	cp = tmp - t_crit
	
	cp = np.where(cp > 0, 0, cp) #Setting all values greater than zero to zero (No Contrails)
	cp = np.where(cp < 0, 1, cp) #Setting all values less than zero to 1 (Contrails)
	
	arr = np.empty(tmp[0,:,:].shape, dtype=float)
	hgt_contrail = np.empty(tmp[0,:,:].shape, dtype=float)
	
	log.info(f"Calculating contrail heights")
	# calc_time = dt.datetime.now()
	for i in range(len(cp[0,:,:])):
		for j in range(len(cp[0,0,:])):
			#print(i)
			#print(j)
			#print(cp[:,i,j])
			idx_total = np.where(cp[:,i,j] == 1)
			#print(idx_total)
			try:
				idx = idx_total[0][0]
			except:
				idx = float("Nan")
				#print("No Contrail Level Found")
			#print(idx)
			arr[i,j] = idx
			try:
				hgtft_val = hgt[idx,i,j]
			except:
				#print("No Contrails Possible")
				hgtft_val = np.nan
			hgt_contrail[i,j] = hgtft_val
			j+=1
		i+=1
	# elapsed_t = dt.datetime.now() - calc_time #Time it took for the script to run
	# elapsed_t = round(elapsed_t.seconds,3) #Time in seconds and rounded to 3 decimal places
	# print (elapsed_t, "seconds.")
	
	# Mask NaNs
	hgt_contrail = np.ma.masked_values([hgt_contrail], np.nan)
	# Make 2d
	hgt_contrail = hgt_contrail[0,:,:]

	tot_time = pd.Timestamp.now(tz='UTC') - now
	log.info(f"Done, took {tot_time.total_seconds()} sec.")

	Calculation = namedtuple('Calculation', ['contrail_heights', 'lat', 'lon', 'init_time', 'valid_time', 'fcst_hr'])

	return Calculation(contrail_heights=hgt_contrail, lat=latitude, lon=longitude, init_time=init_time, valid_time=dt_obj, fcst_hr=fcst_hr)