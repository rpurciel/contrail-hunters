import sys
import os
import glob
from datetime import datetime
import ssl
import urllib.request
import warnings

import xarray as xr
import numpy as np
import pandas as pd

import internal as internal

warnings.filterwarnings("ignore")

#download params
DEF_DL_FORCE_FCST0 = False

def download(save_dir, year, month, day, hour, **kwargs):
    """
    Downloads a single model file to a local directory. 

    Inputs: Directory to save model output files to, and year, month,
            day, and hour to download model file for.

    Returns: Success code, time (in seconds) for function to run,
             path to file
    """
    start_time = datetime.now()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if internal.str_to_bool(kwargs.get('verbose')) == True:
        verbose = True
        print("INFO: VERBOSE mode turned ON")
    else:
        verbose = False

    if internal.str_to_bool(kwargs.get('debug')) == True:
        debug = True
        verbose = True
        print("INFO: DEBUG mode turned ON")
    else:
        debug = False

    if debug:
        print("DEBUG: Kwargs passed:", kwargs)

    if verbose:
        print(f"PROCESSING: Selected time {str(year).zfill(4)}-{str(month).zfill(2)}-{str(day).zfill(2)} {str(hour).zfill(2)}:00:00 UTC")

    force_fcst0 = DEF_DL_FORCE_FCST0
    for arg, value in kwargs.items():
        if arg == 'force_fcst0':
            force_fcst0 = internal.str_to_bool(value)

    if kwargs.get('forecast_hour'):
        forecast_hour = int(kwargs.get('forecast_hour'))
        if verbose:
            print(f"INFO: Downloading for forecast hour {forecast_hour}")

        url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs."+ \
            str(year)+str(month).zfill(2)+str(day).zfill(2)+ \
            "/"+str(hour).zfill(2)+"/atmos/gfs.t"+str(hour).zfill(2)+\
            "z.pgrb2.0p25.f"+str(forecast_hour).zfill(3)
        file_name = "gfs."+str(year)+ str(month).zfill(2) + str(day).zfill(2)+"."+str(hour).zfill(2)+"z.pgrb2.0p25.f"+ \
            str(forecast_hour).zfill(3)
    else:
        if force_fcst0:
            if verbose:
                print(f"INFO: Forcing forecast hour 0 to be downloaded")
            url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs."+ \
                str(year)+str(month).zfill(2)+str(day).zfill(2)+ \
                "/"+str(hour).zfill(2)+"/atmos/gfs.t"+str(hour).zfill(2)+\
                "z.pgrb2.0p25.anl"''
            file_name = "gfs."+str(year)+ str(month).zfill(2) + str(day).zfill(2)+"."+str(hour).zfill(2)+"z.pgrb2.0p25.f000"
        else:
            if verbose:
                print(f"INFO: Downloading analysis data")
            url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs."+ \
                str(year)+str(month).zfill(2)+str(day).zfill(2)+ \
                "/"+str(hour).zfill(2)+"/atmos/gfs.t"+str(hour).zfill(2)+\
                "z.pgrb2.0p25.anl"''
            file_name = "gfs."+str(year)+ str(month).zfill(2) + str(day).zfill(2)+"."+str(hour).zfill(2)+"z.pgrb2.0p25.anl"

    ssl._create_default_https_context = ssl._create_unverified_context

    dest_path = os.path.join(save_dir, file_name)

    if verbose:
        print(f"INFO: Starting downloader...")
    try:
        urllib.request.urlretrieve(url, dest_path) #Retrieve the file and write it as a grbfile
    except urllib.error.URLError as e:

        if verbose:
            print(f"ERROR: {e.reason}")

        elapsed_time = datetime.now() - start_time
        return 0, elapsed_time.total_seconds(), e.reason

    if verbose:
        print(f"INFO: Finished downloading file")

    elapsed_time = datetime.now() - start_time
    return 1, elapsed_time.total_seconds(), dest_path

def plot_plan_view_model(file_path, save_dir, level, variables, points, **kwargs):
    """
    Using a bounding box, plots a list of variables and (opt.) points for a single level.

    Bounding box is specified via a tuple:
    (ll corner lat, ll corner lon, ur corner lat, ur corner lon)

    Variables are specified via a list of short or long names:
    ['t', 'u', 'v', ...] OR ['temp', 'uwind', 'vwind', ...]
    -> These are then compared to a dictionary that converts
       into the appropriate short name for the model at hand.

    Points are specified as a list of tuples, including label:
    [(lat1, lon1, label1), (lat2, lon2, label2), ...]

    Level is specified as a level in hPa, and is interpolated to the nearest
    model level if that exact level is not allowed in the modelling.
    % (TODO: add ability to interpolate to specified level)

    Returns: Success code, time (in seconds) for function to run,
             path to file
    """
    start_time = datetime.now()

    elapsed_time = datetime.now() - start_time
    return 1, elapsed_time.total_seconds(), dest_path

def plot_cross_section_model(file_path, save_dir, start_point, end_point, variables, points, top_level, **kwargs):
    """
    Using a start point & end point, plot variables and (opt.) points (as vlines) to a top level.

    Start and end points are specified as a tuple:
    (lat, lon)

    Variables are specified via a list of short or long names:
    ['t', 'u', 'v', ...] OR ['temp', 'uwind', 'vwind', ...]
    -> These are then compared to a dictionary that converts
       into the appropriate short name for the model at hand.

    Points are specified as a list of tuples, including label:
    [(lat1, lon1, label1), (lat2, lon2, label2), ...]
    * If these points do not have a lat/lon that is exactly
      on the line of the cross-section, they will not be
      plotted.
    % (TODO: add ability to plot points even if off the line)

    Top level is specified as a level in hPa. Default is 100 hPa,
    and it cannot be set lower than the bottom level.

    (Bottom level is by default set to 1000 hPa, but can be modified
    via kwarg bot_level=xxxx.)    

    Returns: Success code, time (in seconds) for function to run,
             path to file
    """
    start_time = datetime.now()

    elapsed_time = datetime.now() - start_time
    return 1, elapsed_time.total_seconds(), dest_path

def raob_csv_sounding_gfs(file_path, save_path, sounding_lat, sounding_lon, **kwargs):
    """
    Using a lat/lon, generates a CSV sounding for use in RAOB.

    Latitude and longitude are specified as decimal floats, with
    the default value set in this file (if none are specified).

    Returns: Success code, time (in seconds) for function to run,
             path to output file
    """

    start_time = datetime.now()

    if kwargs.get('verbose') == True:
        verbose = True
        print("INFO: VERBOSE mode turned ON")
    else:
        verbose = False

    if kwargs.get('debug') == True:
        debug = True
        verbose = True
        print("INFO: DEBUG mode turned ON")
    else:
        debug = False

    if debug:
        print("DEBUG: Kwargs passed:", kwargs)

    if sounding_lon < 0 :
        sounding_lon += 360

    if debug:
        print("DEBUG: Sounding longitude corrected")
        print(f"DEBUG: Original={sounding_lon - 360} New={sounding_lon}")

    ds = xr.open_dataset(file_path, engine='cfgrib',
                        backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa'},'errors':'ignore'})

    timestr = str(pd.to_datetime(ds['time'].values).strftime("%Y-%m-%d %H:%M:00"))
    timestr_file = str(pd.to_datetime(ds['time'].values).strftime("%Y%m%d_%H"))
    fcsthrstr = str(ds['step'].values.astype('timedelta64[h]')).replace(" hours", "")

    # Becuase the grib reader reads longitude from 0-360 and not -180-180
    # we have to adjust the `sounding_lon`.
    if sounding_lon < 0 :
        sounding_lon += 360 

    point_ds = ds.sel(longitude=sounding_lon, latitude=sounding_lat, method='nearest')

    if verbose:
        print("UPPER AIR: Starting interpolation...")
        print('INFO: Requested pt:', sounding_lat, sounding_lon)
        print('INFO: Nearest pt:', point_ds.latitude.data, point_ds.longitude.data)

    latitude_float = round(float(point_ds.latitude.data),2) #Converts the array to a float and rounds it to stick into dataframe later
    longitude_float = round(float(point_ds.longitude.data - 360),2) #Same as for longitude

    press = point_ds.isobaricInhPa.data
    tmp = point_ds.t.data
    hgt = point_ds.gh.data 
    
    #Convert Kelvin temps to C
    tmpc = tmp -273.15

    dsw = xr.open_dataset(file_path, engine='cfgrib',
                        backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa', 'units': 'm s**-1'},'errors':'ignore'})

    point_dsw = dsw.sel(longitude=sounding_lon, latitude=sounding_lat, method= 'nearest')

    if verbose:
        print("UPPER AIR WINDS: Starting interpolation...")
        print('INFO: Requested pt:', sounding_lat, sounding_lon)
        print('INFO: Nearest pt:', point_dsw.latitude.data, point_dsw.longitude.data)

    uwind = point_dsw.u.data
    vwind = point_dsw.v.data
    
    #Convert u & v winds (m/s) to kts
    uwindkts = np.dot(uwind, 1.94384)
    vwindkts = np.dot(vwind, 1.94384)

    dsrh = xr.open_dataset(file_path, engine='cfgrib',
                        backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa', 'shortName': 'r'}})

    point_dsrh = dsrh.sel(longitude=sounding_lon, latitude=sounding_lat, method= 'nearest')

    if verbose:
        print("UPPER AIR RH: Starting interpolation...")
        print('INFO: Requested pt:', sounding_lat, sounding_lon)
        print('INFO: Nearest pt:', point_dsrh.latitude.data, point_dsrh.longitude.data)

    relativehumidity = point_dsrh.r.data
    
    rhappend = relativehumidity
    rhappend = np.append(relativehumidity,0)
    rhappend = np.append(rhappend,0)
    
    #Convert RH to Dewpoint Temp
    A = 17.27
    B = 237.7
    
    #dwptc = B * (np.log(rhappend/100.) + (A*tmpc/(B+tmpc))) / (A-np.log(rhappend/100.)-((A*tmpc)/(B+tmpc)))
    dwptc = B * (np.log(relativehumidity/100.) + (A*tmpc/(B+tmpc))) / (A-np.log(relativehumidity/100.)-((A*tmpc)/(B+tmpc)))

    dsur = xr.open_dataset(file_path, engine='cfgrib',
                    backend_kwargs={'filter_by_keys':{'typeOfLevel': 'surface', 'stepType' : 'instant'},'errors':'ignore'})

    point_dsur = dsur.sel(longitude=sounding_lon, latitude=sounding_lat, method= 'nearest')

    if verbose:
        print("SURFACE: Starting interpolation...")
        print('INFO: Requested pt:', sounding_lat, sounding_lon)
        print('INFO: Nearest pt:', point_dsur.latitude.data, point_dsur.longitude.data)

    presssurface = point_dsur.sp.data
    try:
        tmpsurface = point_dsur.t.data
    except:
        tmpsurface = point_dsur.unknown.data #sometimes the grib reader cant read the temp data?
    hgtsurface = point_dsur.orog.data
    
    #Convert Kelvin temps to C
    tmpsurfacec = tmpsurface -273.15
    
    #Convert Pa Pressure to hpa
    presssurfacePa = presssurface * 0.01
    
    ####Redo Process for 2m Above Ground Level####
    
    #Open the grib2 file with xarray and cfgrib
    
    dg = xr.open_dataset(file_path, engine='cfgrib',
                          backend_kwargs={'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}})

    point_dg = dg.sel(longitude=sounding_lon, latitude=sounding_lat, method= 'nearest')

    if verbose:
        print("SURFACE 2M: Starting interpolation...")
        print('INFO: Requested pt:', sounding_lat, sounding_lon)
        print('INFO: Nearest pt:', point_dg.latitude.data, point_dg.longitude.data)
    
    # These are the profiles you want...
    tmp2m = point_dg.t2m.data
    #d2m = point_dg.d2m.data
    rh2m = point_dg.r2.data
    
    #Convert Kelvin temps to C
    tmp2mc = tmp2m -273.15
    #d2mc = d2m -273.15
    
    d2mc = B * (np.log(rh2m/100.) + (A*tmp2mc/(B+tmp2mc))) / (A-np.log(rh2m/100.)-((A*tmp2mc)/(B+tmp2mc)))
    
    press2mPa = presssurfacePa - .2
    hgt2m = hgtsurface + 2.0
    
    d10m = xr.open_dataset(file_path, engine='cfgrib',
                          backend_kwargs={'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':10}})

    point_d10m = d10m.sel(longitude=sounding_lon, latitude=sounding_lat, method= 'nearest')
    
    if verbose:
        print("SURFACE 10M: Starting interpolation...")
        print('INFO: Requested pt:', sounding_lat, sounding_lon)
        print('INFO: Nearest pt:', point_d10m.latitude.data, point_dg0m.longitude.data)
    
    # These are the profiles you want...
    uwind_10m = point_d10m.u10.data
    vwind_10m = point_d10m.v10.data
    
    #Convert u & v winds (m/s) to kts
    uwindkts_10m = np.dot(uwind_10m, 1.94384)
    vwindkts_10m = np.dot(vwind_10m, 1.94384)
    
    #Create and combine separate data frames in main data frame
    df = pd.DataFrame(data=[press, tmpc, dwptc, uwindkts, vwindkts, hgt])
    #d1 = pd.DataFrame(data=[presssurfacePa,tmpsurfacec,-999,'','',hgtsurface.item(0)]) #Not using surface data becasue no dewpoint present at the surface
    d2 = pd.DataFrame(data=[press2mPa,tmp2mc,d2mc.item(0),uwindkts_10m,vwindkts_10m,hgt2m])
    #d1 = d1.T
    d2 = d2.T
    #df2 = pd.concat([d1,d2],axis=0)
    df_t = df.T
    main_df= pd.concat([d2,df_t],axis=0,ignore_index=True)
    
    #Fill Nan arrays with -999 and get rid of Nan arrays for U & V Columns
    main_df[2].fillna(-999, inplace=True)
    main_df[3].fillna('', inplace=True)
    main_df[4].fillna('', inplace=True)
    
    #Removes the pressure layers below the surface of the ground
    main_df = main_df[main_df[0] <= presssurfacePa]
        
    main_df = main_df.round(decimals=2)
    elev = round(float(hgtsurface)) #Rounding surface elevation for dataframe

    if kwargs.get('sounding_title'):
        csv_name = kwargs.get('sounding_title')
        file_name = date.replace(":","_").replace("-", "_") + "_" + csv_name + "_GFS_RAOB.csv"
    else:
        csv_name = "UNNAMED SOUNDING"
        file_name = date.replace(":","_").replace("-", "_") + "_GFS_RAOB.csv"
    
    d = {0:['RAOB/CSV','DTG','sounding_lat','sounding_lon','ELEV','MOISTURE','WIND','GPM','MISSING','RAOB/DATA','PRES'],
         1:[csv_name,timestr,latitude_float,longitude_float,elev,'TD','kts','MSL',-999,'','TEMP'],2:['','','N','W','m','','U/V','','','','TD'],3:['','','','','','','','','','','UU'],
         4:['','','','','','','','','','','VV'],5:['','','','','','','','','','','GPM']}
    df_2 = pd.DataFrame(data=d)
    
    main_df = pd.concat([df_2,main_df],axis=0,ignore_index=True) #Combines the RAOB Header Format with the sounding data

    dest_path = os.path.join(save_path, file_name)

    main_df.to_csv(dest_path, index=False, header=False)

    if verbose:
        print("FILE: Saved File: " + file_name + " to " + save_path)

    elapsed_time = datetime.now() - start_time
    return 1, elapsed_time.total_seconds(), dest_path

def _calculate_variable_model(variable_short_name, input_data):
    """Internal function: given a short name, return a calculated variable using input data.

    This works strictly with defined calculations, and only with standard (as defined
    for this library) units.
    """

def _convert_natural_name_to_short_name_model(natural_name):
    """Internal function: given a natural variable name, return a short name that works for the specified model.

    This works with certain natural names, and only for defined variables.
    """

def time_remaining_calc(tot_items, processed_items, proc_times_list):

    if processed_items <= 1:
        avg_time = 0
    else:
        avg_time = sum(proc_times_list) / len(proc_times_list)

    time_remaining = (avg_time * (tot_items - processed_items))/3600 #in hours
    tr_hours = time_remaining
    tr_minutes = (time_remaining*60) % 60
    tr_seconds = (time_remaining*3600) % 60

    time_remaining_str = "{}:{}:{}".format(str(round(tr_hours)).zfill(2), str(round(tr_minutes)).zfill(2), str(round(tr_seconds)).zfill(2))

    return time_remaining_str

def total_time_calc(total_time_seconds):

    t_hours = total_time_seconds/3600
    t_minutes = (t_hours*60) % 60
    t_seconds = (t_hours*3600) % 60

    time_str = "{}:{}:{}".format(str(round(t_hours)).zfill(2), str(round(t_minutes)).zfill(2), str(round(t_seconds)).zfill(2))

    return time_str

if __name__ == "__main__":

    # input_save_dir = "/Users/rpurciel/Documents/Voltitude/GFS no ANL"

    # input_year = 2023

    # input_month = 9

    # input_days = range(8, 22, 1)

    # input_hour = range(0, 24, 6)

    # for day in input_days:

    #     for hour in input_hour:

    #         print(day, hour)
    #         download_gfs(input_save_dir, input_year, input_month, day, hour, forecast_hour='0')

    gfs_dir = "/Users/rpurciel/Documents/Voltitude/GFS no ANL/"

    input_save_dir = "/Users/rpurciel/WeatherExtreme Ltd Dropbox/Ryan Purciel/Voltitude/Dropsonde - RAOB/Comparison with GFS/Raw GFS"

    sonde_lats = (16.4649199, 15.9492216, 
        16.6780999, 17.2468933, 17.5750033, 
        17.5512033, 16.2812833, 16.7492233, 
        16.1857866)

    sonde_lons = (-25.45447, -37.589945, 
        -31.22682, -26.75401, -26.499205, 
        -25.2049534, -25.9776217, -26.1555684, 
        -25.6654517)

    sonde_hrs = (18, 0, 6, 18, 18, 12, 18, 18, 18)

    sonde_ids = ('11DS28', '17DS14', 
        '17DS09', ' 17DS08', '13DS03',
        '13DS02', '16DS13', '15DS22',
        '11DS29')

    sonde_dates = (8, 21, 20, 19, 18, 18, 15, 13, 8)

    num_files = 0
    tot_time = 0
    plot_times = []

    tot_files = len(sonde_dates)

    for lat, lon, hr, date, name in zip(sonde_lats, sonde_lons, sonde_hrs, sonde_dates, sonde_ids):

        fname = "gfs.202309" + str(date).zfill(2) + "." + str(hr).zfill(2) + "z.pgrb2.0p25.f000"

        file = os.path.join(gfs_dir, fname)

        status, time, path = raob_csv_sounding_gfs(file, input_save_dir, lat, lon, sounding_title=name, debug=False)

        num_files += status
        tot_time += time
        plot_times += [time]

        time_remaining = time_remaining_calc(tot_files, num_files, plot_times)

        print(f'{num_files}/{tot_files} soundings created. Est. time remaining: {time_remaining}', end='\r')

    tot_time_str = total_time_calc(tot_time)

    print(f'Finished creating {num_files} soundings. Total time: {tot_time_str}         ')






