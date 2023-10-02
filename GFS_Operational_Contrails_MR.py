#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:46:57 2022

@author: mikhailk

5/3/2023 - updated pathnames -mrob
"""

import sys, io, os, shutil, glob
import cv2
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
from PIL import Image

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

from herbie import Herbie
from herbie import FastHerbie

import metpy
from metpy.units import units
import metpy.calc as mpcalc

# Supress warnings
import warnings
warnings.filterwarnings("ignore")

start_time = dt.datetime.now()
print('\n====================================================================')
print ("Script started at ", start_time)
print('====================================================================\n')

# INPUTS
#########################################################

#mainpath
mainpath = "/Users/matthewroberts/Desktop/ScriptTestingEnvs/contrails_MR/"
# Where to save data files
datadir = "/Users/matthewroberts/Desktop/ScriptTestingEnvs/contrails_MR/data/"
# Where to save plots
plotpath = "/Users/matthewroberts/Desktop/ScriptTestingEnvs/contrails_MR/plots/"
# Where to save loops
moviepath = "/Users/matthewroberts/Desktop/ScriptTestingEnvs/contrails_MR/animations/"
# How many hours to plot
hours = 3

# For making colormap
colors = ('red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'violet', 'purple')

#########################################################

# Remove old data before populating with new data
if (os.path.exists(datadir+'gfs/')):
    shutil.rmtree(datadir+'gfs/')
else:
    pass

# if model initialization is <3 hrs old try previous run
if ((pd.Timestamp.utcnow() - pd.Timestamp.utcnow().floor('6H')).seconds/3600 > 3):
    cycle_obj = pd.Timestamp.utcnow().floor('6H')
    cycle_str = cycle_obj.strftime("%Y-%m-%d %H:00") #change the value in the floor function to choose a different run. Sometimes 6H can cause an error if data isn't available, but 12H almost always works.
else:
    cycle_obj = (pd.Timestamp.utcnow().floor('6H') - pd.Timedelta(hours=6))
    cycle_str = cycle_obj.strftime("%Y-%m-%d %H:00")

# Calc forecast hours to plot based on interval and input hours
interval = 3
fxx = range(0, hours+interval, interval)

print('Downloading ' + cycle_str + ' UTC model run...')

# # Download the data
FH = FastHerbie(DATES=[cycle_str], model="gfs", product="pgrb2.0p25", fxx=fxx)
files = FH.download("(?:TMP|HGT|RH):\d+ mb", save_dir=datadir)

#%%
print('Files downloaded. Parsing data...')
# Start opening files and plotting
for path in files:
    
    #Open the grib2 files with xarray and cfgrib
    # Temperature
    dtemp = xr.open_dataset(path, engine='cfgrib', 
                            backend_kwargs={'filter_by_keys':{'cfName': 'air_temperature','typeOfLevel': 'isobaricInhPa'},'errors':'ignore'})                         
    # Height
    dhgt = xr.open_dataset(path, engine='cfgrib', 
                           backend_kwargs={'filter_by_keys':{'cfName': 'geopotential_height','typeOfLevel': 'isobaricInhPa'},'errors':'ignore'})
    # Relative humidity
    dsrh = xr.open_dataset(path, engine='cfgrib', 
                           backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa', 'shortName': 'r'}})

    latitude = dtemp.latitude.data
    longitude = dtemp.longitude.data
    #Valid time for the Forecast Model in UTC
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
    
    print('Plotting...')
    
    # regions = ["CONUS","NA","SA","EUROPE","AFRICA","ASIA","OCEANIA"]
    
    regions = ["ASIA"] #For Testing 
    
    long,lat = np.meshgrid(longitude,latitude) #Making the Lat/Long Arrays 2D
    
    for region in regions:
        
        # Create the figure
        fig = plt.figure(figsize=(15,12)) # (width,height)
        
        ###### TRY THIS CODE #####
        
        # fig,ax = plt.subplots(111, figsize=(15,12), )
        
        ##########################
        
        
        #ax = fig.add_subplot(111)
        # logo = mainpath+"ancillary/Logo.png"
        
        if region == "CONUS":
            #ax = plt.axes(projection = crs.PlateCarree())
            
            ax = plt.axes(projection = crs.LambertConformal(central_longitude = -95, 
                                                      central_latitude = 35,
                                                      standard_parallels = (30,60)))
        
            extent = [-122, -73, 23, 50]
            #extent = [-127, -65, 22, 51]
            
            # Download and add the states, coastlines, counties
            # states = NaturalEarthFeature(category="cultural", scale="50m",
            #                               facecolor="none",
            #                               name="admin_1_states_provinces_shp")
            # ax.add_feature(states, linewidth=3.0, edgecolor="black")
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            # # Adding in Contrail Hunters Logo 
            # img = plt.imread(logo)
            # plt.figimage(img, 6209, 786, zorder=1, alpha=0.6)
            
        if region == "NA":
            ax = plt.axes(projection = crs.LambertConformal(central_longitude = -95, 
                                                      central_latitude = 35))
        
            extent = [-160, -45, 6, 72]

            # Download and add the states, coastlines, counties
            # states = NaturalEarthFeature(category="cultural", scale="50m",
            #                               facecolor="none",
            #                               name="admin_1_states_provinces_shp")
            # ax.add_feature(states, linewidth=2.0, edgecolor="black")
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            # # Adding in Contrail Hunters Logo 
            # img = plt.imread(logo)
            # plt.figimage(img, 6198, 800, zorder=1, alpha=0.6)
            
        # if region == "NA":
        #     ax = plt.axes(projection = crs.PlateCarree())
        
        #     extent = [-170, -45, 6, 72]

        #     # Download and add the states, coastlines, counties
        #     states = NaturalEarthFeature(category="cultural", scale="50m",
        #                                   facecolor="none",
        #                                   name="admin_1_states_provinces_shp")
        #     ax.add_feature(states, linewidth=2.0, edgecolor="black")
        #     ax.set_extent(extent, crs=ccrs.PlateCarree())
            
        #     # Adding in Contrail Hunters Logo 
        #     img = plt.imread(logo)
        #     plt.figimage(img, 6200, 582, zorder=1, alpha=0.6)
            
        if region == "SA":
            ax = plt.axes(projection = crs.PlateCarree())
        
            extent = [-90, -28, -60, 14]
            ax.set_extent(extent, crs=ccrs.PlateCarree())    
            
            # # Adding in Contrail Hunters Logo 
            # img = plt.imread(logo)
            # plt.figimage(img, 4256, 180, zorder=1, alpha=0.6)
            
        if region == "EUROPE":
            ax = plt.axes(projection = crs.PlateCarree())
        
            extent = [-20, 45, 33, 75]
            ax.set_extent(extent, crs=ccrs.PlateCarree())    
            
            # # Adding in Contrail Hunters Logo 
            # img = plt.imread(logo)
            # plt.figimage(img, 6213, 757, zorder=1, alpha=0.6)
            
        if region == "AFRICA":
            ax = plt.axes(projection = crs.PlateCarree())
        
            extent = [-20, 61, -37, 40]
            #extent = [-20, 45, -37, 40]
            ax.set_extent(extent, crs=ccrs.PlateCarree())    
            
            # # Adding in Contrail Hunters Logo 
            # img = plt.imread(logo)
            # plt.figimage(img, 5422, 182, zorder=1, alpha=0.6)
            
        if region == "ASIA":
            ax = plt.axes(projection = crs.PlateCarree())
        
            extent = [40, 155, 0, 85]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            # # Adding in Contrail Hunters Logo 
            # img = plt.imread(logo)
            # plt.figimage(img, 6207, 775, zorder=1, alpha=0.6)
            
        if region == "OCEANIA":
            ax = plt.axes(projection = crs.PlateCarree())
        
            extent = [105, 180, -50, 0]
            ax.set_extent(extent, crs=ccrs.PlateCarree()) 
            
            # # Adding in Contrail Hunters Logo 
            # img = plt.imread(logo)
            # plt.figimage(img, 6207, 775, zorder=1, alpha=0.6)
            
        if region == "SA" or region == "AFRICA":
            hgt_contrail_m = hgt_contrail / 3.281
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
            heights = ax.contourf(long, lat, hgt_contrail_m, levels = levels, cmap = palette, extend = "both", vmin = vmin, vmax = vmax, alpha = .7, transform=ccrs.PlateCarree())#, transform_first = True)
        
        heights = ax.contourf(long, lat, hgt_contrail, levels = levels, cmap = palette, extend = "both", vmin = vmin, vmax = vmax, alpha = .7, transform=ccrs.PlateCarree())#, transform_first = True)
        
        # Add lat/lon grids
        gl = ax.gridlines(linewidth=.8, linestyle='--', color='gray', alpha=.5, draw_labels=True, x_inline=False, y_inline=False, zorder=11)
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 7}
        gl.xlines = True
        gl.ylines = True
        gl.top_labels = False 
        gl.right_labels = False 
        
        #Add borders/coastlines
        ax.coastlines('50m', linewidth=1.5, color='k', alpha=.7, zorder=10)
        ax.add_feature(cfeat.BORDERS, linewidth=1.5, color='k', alpha=.7, zorder=10)
        # ax.add_feature(cartopy.feature.LAKES.with_scale('10m'), linestyle='-', linewidth=0.5, alpha=1,edgecolor='blue',facecolor='none')
        
        # Master city file
        final_stalist = pd.read_csv(mainpath+"ancillary/LargeCities.csv", sep=',', header=0)

        # Lists for the lat & long coordinates for points to be plotted on the map.
        label_stalist = final_stalist.loc[final_stalist['lat'] < extent[3]]
        label_stalist = label_stalist.loc[label_stalist['lat'] > extent[2]]
        label_stalist = label_stalist.loc[label_stalist['lng'] < extent[1]] 
        label_stalist = label_stalist.loc[label_stalist['lng'] > extent[0]]
        
        x_axis_points = label_stalist.lng.to_numpy()
        y_axis_points = label_stalist.lat.to_numpy()
        labels = label_stalist.city_ascii.to_numpy()
        lab_xpoints = x_axis_points
        lab_ypoints = y_axis_points
        
        """
        ################# NOT WORKING #################
        # lab_xpoints = []
        # lab_ypoints = []
        # # Fix city labels that are too close or overlapping
        # for pt in range(len(x_axis_points)):
        #     ydist = np.abs(y_axis_points[pt]) - np.abs(y_axis_points)
        #     xdist = np.abs(x_axis_points[pt]) - np.abs(x_axis_points)
        #     # Where are the labels too close?
        #     idx = np.where(((ydist > 0) & (ydist < 2)) & (xdist < 10))
        #     # If points are too close move label a bit in opposite direction
        #     # print(lab_ypoints[pt])
        #     if (len(idx[0]) > 0):
        #         # print('yes')
        #         ypt = y_axis_points[pt] + value
        #         xpt = x_axis_points[pt] + value
        #     else:
        #         # print('no')
        #         ypt = y_axis_points[pt] + value
        #         xpt = x_axis_points[pt] + value
            
        #     lab_ypoints.append(ypt)
        #     lab_xpoints.append(xpt)
        
        # lab_ypoints = np.asarray(lab_ypoints)
        # lab_xpoints = np.asarray(lab_xpoints)
        # # print(lab_ypoints[pt])
        
        ###################################################
        """
        
        value = .44 # Padding for text labels
        
        # Plot city points/labels
        for (g,h,i,j,k) in zip(lab_xpoints, lab_ypoints, x_axis_points, y_axis_points, labels):
        
            # Plotting the points
            pts = ax.scatter(i, j, color='white', zorder=12,
                              s=90, marker='*', transform=crs.PlateCarree())
            pts = ax.scatter(i, j, color='k', zorder=12,
                              s=60, marker='*', transform=crs.PlateCarree())

            labi = g + value
            labj = h + value
            # Adding Text Label for the points
            txt = ax.text(labi, labj, k, zorder=12,
                           horizontalalignment='left', color='k', fontsize=7, alpha=.9,
                           fontweight = 'semibold', transform=ccrs.PlateCarree())
            txt.set_path_effects([PathEffects.withStroke(linewidth=.5, foreground='w')])
            
        # Plot title and headers
        ttl_init_str = cycle_obj.strftime("%Hz %b %d %Y")
        fcasthr = str(path)[-3:]
        ttl_valid_str = dt_obj.strftime("%Hz %b %d %Y")
        ax.set_title(r"$\bf{"+"GFS\ Minimum\ Contrail\ Heights\ (ft\ &\ m\ MSL)"+"}$"+"\nInit: "+ttl_init_str+"   Forecast Hr: ["+fcasthr+"]   Valid: "+ttl_valid_str, 
                  loc='left', fontsize = 13)
        ax.set_title(r"ContrailHunters.com", color='gray', fontweight='bold', fontsize=13, loc='right')
        
        # Plot colorbar
        fig.colorbar(heights, shrink=.5, aspect=22, pad=.01, extendfrac=.02)
        
        # Add logo to corner
        logo = mainpath+"ancillary/Logo.png"
        img = Image.open(logo)
        im_width, im_height = img.size
        margin_x, margin_y = 140, 80
        fig.figimage(img, xo=fig.bbox.xmin + margin_x, yo=fig.bbox.ymin + margin_y, zorder=15, alpha=0.6)
        
        
        """
        # Making colorbar axes ###### NEEDS CHANGES ######
        if region == "CONUS" or region == "NA" or region == "EUROPE":
            cax12 = plt.axes([0.125, 0.125, 0.775, 0.05]) #Left, bottom, width, height
        
        if region == "ASIA":
            cax12 = plt.axes([0.125, 0.08, 0.775, 0.05]) #Left, bottom, width, height
            
        if region == "OCEANIA":
            cax12 = plt.axes([0.125, 0.113, 0.775, 0.05]) #Left, bottom, width, height
        
        if region == "SA":
            cax12 = plt.axes([0.83, 0.125, 0.033, 0.75]) #Left, bottom, width, height
        
        if region == "AFRICA":
            cax12 = plt.axes([0.90, 0.125, 0.033, 0.755]) #Left, bottom, width, height
        
        if region == "CONUS" or region == "NA" or region == "EUROPE" or region == "ASIA" or region == "OCEANIA":
            #Creating the 2 Colorbar Axes    
            cax3 = cax12.twiny()
            ticks = [4572,6096,7620,9144,10668,12192,13716,15240,16764]
            cbar = plt.colorbar(heights, cax=cax12, orientation="horizontal", pad=.03)
            cbar.ax.set_xticklabels(['15000','20000','25000','30000','35000','40000','45000','50000','55000'])
            cbar.ax.tick_params(labelsize=20)
            cax3.set_xlim(4572,16764)
            cax3.set_xticks(ticks)
            cax3.tick_params(labelsize=20)
        
        if region == "SA" or region == "AFRICA":
            #Creating the 2 Colorbar Axes    
            cax3 = cax12.twinx()
            ticks = [15000,20000,25000,30000,35000,40000,45000,50000,55000]
            cbar = plt.colorbar(heights, cax=cax12, orientation="vertical", pad=.03)
            cbar.ax.set_yticklabels(['4572','6096','7620','9144','10668','12192','13716','15240','16764'])
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.yaxis.set_label_position('left')
            cbar.ax.tick_params(labelsize=20)
            cax3.set_ylim(15000,55000)
            cax3.set_yticks(ticks)
            cax3.tick_params(labelsize=20)
        """
        
        print('Saving...', end='', flush=True)
        savename = dt.datetime.strftime(dt_obj,"%Y%m%d_%H%Mz") + "_" + region + ".png"
        plt.savefig(plotpath+"GFS_Contrail_Heights_" + savename, bbox_inches="tight",dpi=200)
        buf = io.BytesIO()
        plt.close()
        buf.seek(0)
        buf.close()
        print(region+" Contrail Heights")
        

# all_files = glob.glob("/Users/mikhailk/Desktop/Aera/NAM/*.idx") #All the files

# for filePath in all_files: #Removes them iteratively 
#     try:
#         os.remove(filePath)
#     except:
#         print("Error while deleting file : ", filePath)
            
elapsed_t = dt.datetime.now() - start_time #Time it took for the script to run
elapsed_t = round(elapsed_t.seconds/60,3) #Time in minutes and rounded to 3 decimal places
print('\n====================================================================')
print ("Script finished in", elapsed_t, "minutes")
print('====================================================================')
    
    
