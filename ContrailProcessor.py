import os
import shutil

from modeldata import ModelRunSeries, SingleRunData

from herbie import Herbie
from herbie import FastHerbie

import pandas as pd
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import from_levels_and_colors, to_rgba
import matplotlib.patheffects as PathEffects

import cartopy
import cartopy.crs as crs
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.util import add_cyclic_point
from cartopy.feature import NaturalEarthFeature

import metpy
from metpy.units import units

def download_current_gfs(forecast_length, forecast_interval, download_directory):

	#model selection params
	model = "gfs"
	product = "pgrb2.0p25"

	download_path = os.path.join(download_directory, model)

	 #create directory if it doesnt exist, and if it does delete old plots
	if os.path.exists(download_path):
		shutil.rmtree(download_path)
	else:	
			os.makedirs(download_path)

		#Get the time string of the closest model run, used to select data to download
	if ((pd.Timestamp.utcnow() - pd.Timestamp.utcnow().floor('6H')).seconds/3600 > 3):
		cycle_obj = pd.Timestamp.utcnow().floor('6H')
		cycle_str = cycle_obj.strftime("%Y-%m-%d %H:00")
	else:
		cycle_obj = (pd.Timestamp.utcnow().floor('6H') - pd.Timedelta(hours=6))
		cycle_str = cycle_obj.strftime("%Y-%m-%d %H:00")

	forecast_hours = range(0, forecast_length + forecast_interval, forecast_interval)

	FH = FastHerbie(DATES=[cycle_str], model=model, 
					product=product, fxx=forecast_hours)
	files = FH.download("(?:TMP|HGT|RH):\d+ mb", save_dir=download_path)

	return ModelRunSeries(model, product, cycle_obj, forecast_length, forecast_interval, files)

def calculate_contrail_heights(SingleRunData):

    data_file_path = SingleRunData['file_path']    
    
    dtemp = xr.open_dataset(data_file_path, engine='cfgrib', 
                            backend_kwargs={'filter_by_keys':{'cfName': 'air_temperature','typeOfLevel': 'isobaricInhPa'},'errors':'ignore'})                         
    # Height
    dhgt = xr.open_dataset(data_file_path, engine='cfgrib', 
                           backend_kwargs={'filter_by_keys':{'cfName': 'geopotential_height','typeOfLevel': 'isobaricInhPa'},'errors':'ignore'})
    # Relative humidity
    dsrh = xr.open_dataset(data_file_path, engine='cfgrib', 
                           backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa', 'shortName': 'r'}})

    latitude = dtemp.latitude.data
    longitude = dtemp.longitude.data
    np_dt_obj = dtemp.valid_time.data
    dt_obj = pd.to_datetime(np_dt_obj)
    dt_str = str(dt_obj)
    
    print('\nParsing '+dt_str+'...', end='', flush=True)
    
    # Pressure, temp, height, RH, specific humidity data
    press = dtemp.isobaricInhPa.data
    tmp = dtemp.t.data-273.15 #Kelvin to degC
    hgt = dhgt.gh.data*3.28084 # GPM meters to feet
    rh = dsrh.r.data
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
    
    print('Determining contrail heights...', end='', flush=True)
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
    
    SingleRunData['contrail_heights'] = hgt_contrail
    SingleRunData['latitude'] = latitude
    SingleRunData['longitude'] = longitude

    return SingleRunData

def plot_region(SingleRunData, region):

    contrail_heights = SingleRunData['contrail_heights']
    latitude = SingleRunData['latutde']
    longitude = SingleRunData['longitude']
    
    long,lat = np.meshgrid(longitude,latitude)
    
    colors = ('red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'violet', 'purple')

    fig = plt.figure(figsize=(15,12))
    
    if region == "CONUS":
        
        ax = plt.axes(projection = crs.LambertConformal(central_longitude = -95, 
                                                  		central_latitude = 35,
                                                  		standard_parallels = (30,60)))
        extent = [-122, -73, 23, 50]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
    if region == "NA":
        ax = plt.axes(projection = crs.LambertConformal(central_longitude = -95, 
                                                  		central_latitude = 35))
        extent = [-160, -45, 6, 72]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
    if region == "SA":
        ax = plt.axes(projection = crs.PlateCarree())
    
        extent = [-90, -28, -60, 14]
        ax.set_extent(extent, crs=ccrs.PlateCarree())    
        
    if region == "EUROPE":
        ax = plt.axes(projection = crs.PlateCarree())
    
        extent = [-20, 45, 33, 75]
        ax.set_extent(extent, crs=ccrs.PlateCarree())    

    if region == "AFRICA":
        ax = plt.axes(projection = crs.PlateCarree())
    
        extent = [-20, 61, -37, 40]
        ax.set_extent(extent, crs=ccrs.PlateCarree())    
        
        
    if region == "ASIA":
        ax = plt.axes(projection = crs.PlateCarree())
    
        extent = [40, 155, 0, 85]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
    if region == "OCEANIA":
        ax = plt.axes(projection = crs.PlateCarree())
    
        extent = [105, 180, -50, 0]
        ax.set_extent(extent, crs=ccrs.PlateCarree()) 
        
    if region == "SA" or region == "AFRICA":
        contrail_heights_m = contrail_heights / 3.281
        levels = [4572,6096,7620,9144,10668,12192,13716,15240,16764]
        vmin = 4572
        vmax = 16764
        
    ## Plotting the Probability Data ##
    levels = [15000,20000,25000,30000,35000,40000,45000,50000,55000]
    vmin = 15000
    vmax = 55000
    #palette = copy(plt.get_cmap("Set1")) #Copying the color map to modify it
    palette, palette_norm = from_levels_and_colors(levels, colors)
    palette.set_under('darkred') #Creating White for anything below 15000
    palette.set_over('navy') #Creating White for anything above 55000
    #palette.set_over('darkred') #Creating White for anything below 15000
    #palette.set_under('navy') #Creating White for anything above 55000
    
    if region == "SA" or region == "AFRICA":
        contrail_heights_plotted = ax.contourf(long, lat, contrail_heights_m, 
        									   levels = levels, cmap = palette, 
        									   extend = "both", vmin = vmin, 
        									   vmax = vmax, alpha = .7, 
        									   transform=ccrs.PlateCarree())
        									   #, transform_first = True)
    
    contrail_heights_plotted = ax.contourf(long, lat, contrail_heights, 
    									   levels = levels, cmap = palette, 
    									   extend = "both", vmin = vmin, 
    									   vmax = vmax, alpha = .7, 
    									   transform=ccrs.PlateCarree())
    									   #, transform_first = True)
    
    # Add lat/lon grids
    gridlines = ax.gridlines(linewidth=.8, linestyle='--', 
    						color='gray', alpha=.5, draw_labels=True, 
    						x_inline=False, y_inline=False, zorder=11)
    gridlines.xlabel_style = {'size': 7}
    gridlines.ylabel_style = {'size': 7}
    gridlines.xlines = True
    gridlines.ylines = True
    gridlines.top_labels = False 
    gridlines.right_labels = False 
    
    ax.coastlines('50m', linewidth=1.5, color='k', alpha=.7, zorder=10)
    ax.add_feature(cfeat.BORDERS, linewidth=1.5, color='k', alpha=.7, zorder=10)
    
    # Master city file
    master_city_list = pd.read_csv(os.path.join(ancillary_path, "LargeCities.csv"), sep=',', header=0)

    # Lists for the lat & long coordinates for points to be plotted on the map.
    selected_city_list = master_city_list.loc[master_city_list['lat'] < extent[3]]
    selected_city_list = selected_city_list.loc[selected_city_list['lat'] > extent[2]]
    selected_city_list = selected_city_list.loc[selected_city_list['lng'] < extent[1]] 
    selected_city_list = selected_city_list.loc[selected_city_list['lng'] > extent[0]]
    
    selected_cities_x_axis_list = selected_city_list.lng.to_numpy()
    selected_cities_y_axis_list = selected_city_list.lat.to_numpy()
    selected_cities_labels_list = selected_city_list.city_ascii.to_numpy()
    # ?? why
    # lab_xpoints = selected_cities_x_axis
    # lab_ypoints = selected_cities_y_axis
    
    text_label_padding = .44 # Padding for text labels
    
    # Plot city points/selected_cities_labels
    for (g,h,i,j,k) in zip(selected_cities_x_axis_list, 
    					   selected_cities_y_axis_list, 
    					   selected_cities_x_axis_list, 
    					   selected_cities_y_axis_list, 
    					   selected_cities_labels_list):
    
        # Plotting the points
        selected_city_points = ax.scatter(i, j, color='white', zorder=12,
                          s=90, marker='*', transform=crs.PlateCarree())
        selected_city_points = ax.scatter(i, j, color='k', zorder=12,
                          s=60, marker='*', transform=crs.PlateCarree())

        selected_city_label_i = g + text_label_padding
        selected_city_label_j = h + text_label_padding

        selected_city_labels = ax.text(selected_city_label_i, 
        			   selected_city_label_j, k, zorder=12,
                       horizontalalignment='left', color='k', fontsize=7, alpha=.9,
                       fontweight = 'semibold', transform=ccrs.PlateCarree())
        selected_city_labels.set_path_effects([PathEffects.withStroke(linewidth=.5, foreground='w')])
        
    # Plot title and headers
    ttl_init_str = cycle_obj.strftime("%Hz %b %d %Y")
    fcasthr = str(path)[-3:]
    ttl_valid_str = dt_obj.strftime("%Hz %b %d %Y")
    plot_title = r"$\bf{"+"GFS\ Minimum\ Contrail\ Heights\ (ft\ &\ m\ MSL)"+"}$"+"\nInit: "+ttl_init_str+"   Forecast Hr: ["+fcasthr+"]   Valid: "+ttl_valid_str
    ax.set_title(plot_title, loc='left', fontsize = 13)
    ax.set_title(r"ContrailHunters.com", color='gray', fontweight='bold', 
                 fontsize=13, loc='right')
    
    fig.colorbar(contrail_heights_plotted, shrink=.5, aspect=22, pad=.01, extendfrac=.02)
    
    # Add logo to corner
    logo = mainpath+"ancillary/Logo.png"
    img = Image.open(logo)
    im_width, im_height = img.size
    margin_x, margin_y = 140, 80
    fig.figimage(img, xo=fig.bbox.xmin + margin_x, yo=fig.bbox.ymin + margin_y, zorder=15, alpha=0.6)
    
    
    print('Saving...', end='', flush=True)
    savename = dt.datetime.strftime(dt_obj,"%Y%m%d_%H%Mz") + "_" + region + ".png"
    plt.savefig(plotpath+"GFS_Contrail_Heights_" + savename, bbox_inches="tight",dpi=200)
    buf = io.BytesIO()
    plt.close()
    buf.seek(0)
    buf.close()
    print(region+" Contrail Heights")