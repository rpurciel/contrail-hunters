import sys
import os
import shutil
from collections import namedtuple
import warnings
import logging
from datetime import datetime

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
from PIL import Image

import util as internal
import config as cfg

warnings.filterwarnings('ignore')

DEF_COLORBAR_VISIBLE = True
DEF_COLORBAR_LABEL_VISIBLE = False
DEF_COLORBAR_LABEL = 'Contrail Heights'
DEF_COLORBAR_SHRINK_FACT = 0.5
DEF_COLORBAR_LOCATION = 'right'
DEF_COLORBAR_TICK_LABEL_FORMAT = "FL{x:.2f}"

DEF_PLOT_TITLE = ""
DEF_PLOT_BRANDING = "ContrailHunters®"

DEF_FILE_DPI = 300

log = logging.getLogger("proc")
log_format = logging.Formatter("[%(asctime)s:%(filename)s:%(lineno)s]%(levelname)s:%(message)s",
                               datefmt="%Y-%m-%d %H:%M:%S %Z",)
# if cfg.DEBUG_LOG_TO_STDOUT:
#     #log_channel = logging.StreamHandler(stream=sys.stdout)

#log_channel.setFormatter(log_format)
#log.addHandler(log_channel)
log.setLevel(eval(cfg.DEBUG_LOGGING_LEVEL))

def calculate_contrail_heights(data_file):

    start_time = datetime.now()

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

    Calculation = namedtuple('Calculation', ['contrail_heights', 'lat', 'lon', 'init_time', 'valid_time', 'fcst_hr'])

    elapsed_time = datetime.now() - start_time
    return Calculation(contrail_heights=hgt_contrail, lat=latitude, lon=longitude, init_time=init_time, valid_time=dt_obj, fcst_hr=fcst_hr), elapsed_time.total_seconds()

def plot_region(save_dir, Calculation, region_id, **kwargs):

    start_time = datetime.now()  

    contrail_heights = Calculation.contrail_heights
    latitude = Calculation.lat
    longitude = Calculation.lon

    init_time = Calculation.init_time
    valid_time = Calculation.valid_time
    fcst_hr = Calculation.fcst_hr
    
    long,lat = np.meshgrid(longitude,latitude)

    fig = plt.figure(figsize=(15,12))
    
    Region = _dynamic_region_selector(region_id)

    extent = Region.extent
    proj = Region.proj
    region_name = Region.name
        
    ax = plt.axes(projection = proj)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
        
    if cfg.DEF_UNITS == "m":
        contrail_heights = contrail_heights / 3.281
        levels = cfg.DEF_METER_LEVELS
        vmin = cfg.DEF_METER_MIN
        vmax = cfg.DEF_METER_MAX
    else: #feet, cfg.DEFault unit
        levels = cfg.DEF_FEET_LEVELS
        vmin = cfg.DEF_FEET_MIN
        vmax = cfg.DEF_FEET_MAX

    #palette = copy(plt.get_cmap("Set1")) #Copying the color map to modify it

    palette, palette_norm = from_levels_and_colors(levels, cfg.DEF_COLOR_PALLETE)
    palette.set_under(cfg.DEF_COLOR_BELOW_MIN) #Creating White for anything below 15000
    palette.set_over(cfg.DEF_COLOR_ABOVE_MAX) #Creating White for anything above 55000
    
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

    if cfg.DEF_GEOG_VISIBLE:
        if cfg.DEF_GEOG_DRAW_STATES:
            ax.add_feature(crs.cartopy.feature.STATES, zorder=10)
            log.debug("Drawing states")
        if cfg.DEF_GEOG_DRAW_COASTLINES:
            ax.coastlines('50m', linewidth=1.5, color='k', alpha=.7, zorder=10)
            log.debug("Drawing coastlines")
        if cfg.DEF_GEOG_DRAW_BORDERS:
            ax.add_feature(cfeat.BORDERS, linewidth=1.5, color='k', alpha=.7, zorder=10)
            log.debug("Drawing borders")
        if cfg.DEF_GEOG_DRAW_WATER:
            ax.add_feature(cfeature.LAKES.with_scale('10m'),linestyle='-',linewidth=0.5,alpha=1,edgecolor='blue',facecolor='none', zorder=10)
            log.debug("Drawing water")
    else:
        log.debug("Geography drawing turned OFF")

    # # Master city file
    # master_city_list = pd.read_csv(os.path.join(ancillary_path, "LargeCities.csv"), sep=',', header=0)

    # # Lists for the lat & long coordinates for points to be plotted on the map.
    # selected_city_list = master_city_list.loc[master_city_list['lat'] < extent[3]]
    # selected_city_list = selected_city_list.loc[selected_city_list['lat'] > extent[2]]
    # selected_city_list = selected_city_list.loc[selected_city_list['lng'] < extent[1]] 
    # selected_city_list = selected_city_list.loc[selected_city_list['lng'] > extent[0]]
    
    # selected_cities_x_axis_list = selected_city_list.lng.to_numpy()
    # selected_cities_y_axis_list = selected_city_list.lat.to_numpy()
    # selected_cities_labels_list = selected_city_list.city_ascii.to_numpy()
    # # ?? why
    # # lab_xpoints = selected_cities_x_axis
    # # lab_ypoints = selected_cities_y_axis
    
    # text_label_padding = .44 # Padding for text labels
    
    # # Plot city points/selected_cities_labels
    # for (g,h,i,j,k) in zip(selected_cities_x_axis_list, 
    # 					   selected_cities_y_axis_list, 
    # 					   selected_cities_x_axis_list, 
    # 					   selected_cities_y_axis_list, 
    # 					   selected_cities_labels_list):
    
    #     # Plotting the points
    #     selected_city_points = ax.scatter(i, j, color='white', zorder=12,
    #                       s=90, marker='*', transform=crs.PlateCarree())
    #     selected_city_points = ax.scatter(i, j, color='k', zorder=12,
    #                       s=60, marker='*', transform=crs.PlateCarree())

    #     selected_city_label_i = g + text_label_padding
    #     selected_city_label_j = h + text_label_padding

    #     selected_city_labels = ax.text(selected_city_label_i, 
    #     			   selected_city_label_j, k, zorder=12,
    #                    horizontalalignment='left', color='k', fontsize=7, alpha=.9,
    #                    fontweight = 'semibold', transform=ccrs.PlateCarree())
    #     selected_city_labels.set_path_effects([PathEffects.withStroke(linewidth=.5, foreground='w')])
        
    # Plot title and headers


    ttl_init_str = init_time.strftime("%H:00 UTC %b %d %Y")
    fcasthr = str(fcst_hr)
    ttl_valid_str = valid_time.strftime("%H:00 UTC %b %d %Y")
    plot_title = r"$\bf{"+"GFS\ Minimum\ Contrail\ Heights\ (ft\ &\ m\ MSL)"+"}$"+"\nInit: "+ttl_init_str+"   Forecast Hr: ["+fcasthr+"]   Valid: "+ttl_valid_str
    ax.set_title(plot_title, loc='left', fontsize = 13)
    ax.set_title(cfg.DEF_PLOT_BRANDING + "\n", color='gray', fontweight='bold', 
                 fontsize=13, loc='right')

    if cfg.DEF_COLORBAR_VISIBLE:
        if cfg.DEF_COLORBAR_LOCATION == 'inside':
            cb = fig.colorbar(contrail_heights_plotted, orientation = "horizontal", shrink=cfg.DEF_COLORBAR_SHRINK_FACT, aspect=22, 
                              pad=-.15, extendfrac=.02, format=cfg.DEF_COLORBAR_TICK_LABEL_FORMAT)
            log.debug("Drawing colorbar turned ON")
            log.debug("Location = Inside plot")
            log.debug(f"Tick Label Format = '{cfg.DEF_COLORBAR_TICK_LABEL_FORMAT}'")
        if cfg.DEF_COLORBAR_LOCATION == 'bottom':
            cb = fig.colorbar(contrail_heights_plotted, orientation = "horizontal", shrink=cfg.DEF_COLORBAR_SHRINK_FACT, aspect=22, 
                              pad=.01, extendfrac=.02, format=cfg.DEF_COLORBAR_TICK_LABEL_FORMAT)
            log.debug("Drawing colorbar turned ON")
            log.debug("Location = Bottom of plot")
            log.debug(f"Tick Label Format = '{cfg.DEF_COLORBAR_TICK_LABEL_FORMAT}'")
        if cfg.DEF_COLORBAR_LOCATION == 'right':
            cb = fig.colorbar(contrail_heights_plotted, orientation = "vertical", shrink=cfg.DEF_COLORBAR_SHRINK_FACT, aspect=22, 
                              pad=.01, extendfrac=.02, format=cfg.DEF_COLORBAR_TICK_LABEL_FORMAT)
            log.debug("Drawing colorbar turned ON")
            log.debug("Location = Left of plot")
            log.debug(f"Tick Label Format = '{cfg.DEF_COLORBAR_TICK_LABEL_FORMAT}'")

        if cfg.DEF_COLORBAR_LABEL_VISIBLE:
            cb.set_label(cfg.DEF_COLORBAR_LABEL)
            log.debug("Colorbar label turned ON")
            log.debug(f"Label = '{cfg.DEF_COLORBAR_LABEL}'")
    
    # Add logo to corner
    cwd = os.getcwd()
    ancillary_dir = os.path.join(cwd, cfg.DEF_ANCILLARY_DIR_NAME)
    logo_sized = f"logo{cfg.DEF_LOGO_SIZE_PIX}.png"
    logo = os.path.join(ancillary_dir, logo_sized)
    img = Image.open(logo)
    # ax.imshow(img, extent=(0.4, 0.6, 5, 5), zorder=15, alpha=cfg.DEF_LOGO_ALPHA)
    fig.figimage(img, xo=ax.bbox.xmin + cfg.DEF_LOGO_MARGIN_X, yo=ax.bbox.ymin + cfg.DEF_LOGO_MARGIN_X, zorder=15, alpha=cfg.DEF_LOGO_ALPHA)

    region_name = region_name.replace(" ", "").upper()
    valid_time_str = valid_time.strftime('%Y%m%d_%H%M%S%Z')
    
    file_name = eval(cfg.DEF_FILE_NAMING_SCHEME)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dest_path = os.path.join(save_dir, file_name + ".png")

    try:
        plt.savefig(dest_path, bbox_inches="tight", dpi=cfg.DEF_FILE_DPI)
    except:
        raise RuntimeError("Could not save file")
    else:
        log.info(f"Sucessfully saved plot file to {dest_path}")
        plt.close()

    # buf = io.BytesIO()
    # plt.close()
    # buf.seek(0)
    # buf.close()

    elapsed_time = datetime.now() - start_time
    return elapsed_time.total_seconds()

def _dynamic_region_selector(sel_id):
    sel_region = {}

    for region in cfg.DEF_REGIONS:
        if region['id'] == sel_id:
            sel_region = cfg.DEF_REGIONS[cfg.DEF_REGIONS.index(region)]

    reg_id = sel_region['id']
    reg_name = sel_region['name']
    reg_bbox = [sel_region['west'], sel_region['east'], sel_region['south'], sel_region['north']] #WESN
    reg_proj = sel_region['proj']
    
    if reg_proj == 'PlateCarree':
        sel_proj = crs.PlateCarree()
    elif reg_proj == 'LambertConformal':
        if 'central_lat' and 'central_lon' in sel_region:
            if 'std_parallels' in sel_region:
                sel_proj = crs.LambertConformal(central_longitude=sel_region['central_lon'],
                                                central_latitude=sel_region['central_lat'],
                                                standard_parallels=sel_region['std_parallels'])
            else:
                sel_proj = crs.LambertConformal(central_longitude=sel_region['central_lon'],
                                                central_latitude=sel_region['central_lat'])
        else:
            sel_proj = crs.LambertConformal()
    else:
        sel_proj = crs.PlateCarree()

    Region = namedtuple('Region', ['id', 'name', 'extent', 'proj'])

    return Region(id=reg_id, name=reg_name, extent=reg_bbox, proj=sel_proj)
