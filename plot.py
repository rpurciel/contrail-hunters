import os
import tomllib
import datetime
import logging
from collections import namedtuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import from_levels_and_colors, to_rgba
import matplotlib.patheffects as PathEffects
import cartopy
import cartopy.crs as crs
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.feature import NaturalEarthFeature
from PIL import Image

PATH_TO_LOGCFG = "config/logging.conf"

with open(os.path.join(os.getcwd(), PATH_TO_LOGCFG), 'rb') as log_cfg:
	logging.config.dictConfig(tomllib.load(log_cfg))

global log
log = logging.getLogger("main.plot")

def plot_region(save_dir: str,
				Calculation: namedtuple,
				region_id: str,
				config: dict):

	now = pd.Timestamp.now(tz='UTC')
	log.info(f"Starting plotting routine for region ID {region_id} at {now} UTC")

	contrail_heights = Calculation.contrail_heights
	latitude = Calculation.lat
	longitude = Calculation.lon

	init_time = Calculation.init_time
	valid_time = Calculation.valid_time
	fcst_hr = Calculation.fcst_hr
	
	long,lat = np.meshgrid(longitude,latitude)

	fig = plt.figure(figsize=(15,12))

	region_cfg_path = os.path.join(os.getcwd(), config['misc']['RegionConfigPath'])
	with open(region_cfg_path, 'rb') as region_cfg:
		region_info = tomllib.load(region_cfg)
	
	Region = _dynamic_region_selector(region_id, region_info)

	extent = Region.extent
	proj = Region.proj
	region_name = Region.name

	log.info(f"Plotting region {region_name}")
		
	ax = plt.axes(projection = proj)
	ax.set_extent(extent, crs=ccrs.PlateCarree())
		
	if config['calculation']['Units'] == "m":
		contrail_heights = contrail_heights / 3.281
		levels = config['plotting']['units']['mLevels']
		vmin = config['plotting']['units']['mMinLevel']
		vmax = config['plotting']['units']['mMaxLevel']
	else: #feet, default unit
		levels = config['plotting']['units']['ftLevels']
		vmin = config['plotting']['units']['ftMinLevel']
		vmax = config['plotting']['units']['ftMaxLevel']

	#palette = copy(plt.get_cmap("Set1")) #Copying the color map to modify it

	palette, palette_norm = from_levels_and_colors(levels, tuple(config['plotting']['colors']['ColorPallete']))
	palette.set_under(config['plotting']['colors']['BelowMinHeightColor']) #Creating White for anything below 15000
	palette.set_over(config['plotting']['colors']['AboveMaxHeightColor']) #Creating White for anything above 55000
	
	contrail_heights_plotted = ax.contourf(long, lat, contrail_heights, 
										   levels = levels, cmap = palette, 
										   extend = "both", vmin = vmin, 
										   vmax = vmax, alpha = .7, 
										   transform=ccrs.PlateCarree(),
										   transform_first = True)
	
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

	if config['plotting']['geography']['Visible']:
		if config['plotting']['geography']['DrawStates']:
			ax.add_feature(crs.cartopy.feature.STATES, zorder=10)
			log.debug("Drawing states")
		if config['plotting']['geography']['DrawCoastlines']:
			ax.coastlines('50m', linewidth=1.5, color='k', alpha=.7, zorder=10)
			log.debug("Drawing coastlines")
		if config['plotting']['geography']['DrawBorders']:
			ax.add_feature(cfeature.BORDERS, linewidth=1.5, color='k', alpha=.7, zorder=10)
			log.debug("Drawing borders")
		if config['plotting']['geography']['DrawWater']:
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
	
	#	 # Plotting the points
	#	 selected_city_points = ax.scatter(i, j, color='white', zorder=12,
	#					   s=90, marker='*', transform=crs.PlateCarree())
	#	 selected_city_points = ax.scatter(i, j, color='k', zorder=12,
	#					   s=60, marker='*', transform=crs.PlateCarree())

	#	 selected_city_label_i = g + text_label_padding
	#	 selected_city_label_j = h + text_label_padding

	#	 selected_city_labels = ax.text(selected_city_label_i, 
	#	 			   selected_city_label_j, k, zorder=12,
	#					horizontalalignment='left', color='k', fontsize=7, alpha=.9,
	#					fontweight = 'semibold', transform=ccrs.PlateCarree())
	#	 selected_city_labels.set_path_effects([PathEffects.withStroke(linewidth=.5, foreground='w')])
		
	# Plot title and headers


	ttl_init_str = init_time.strftime("%H:00 UTC %b %d %Y")
	fcasthr = str(fcst_hr)
	ttl_valid_str = valid_time.strftime("%H:00 UTC %b %d %Y")
	plot_title = r"$\bf{"+"GFS\ Minimum\ Contrail\ Heights\ (ft\ &\ m\ MSL)"+"}$"+"\nInit: "+ttl_init_str+"   Forecast Hr: ["+fcasthr+"]   Valid: "+ttl_valid_str
	ax.set_title(plot_title, loc='left', fontsize = 13)
	ax.set_title(config['plotting']['Branding'] + "\n", color='gray', fontweight='bold', 
				 fontsize=13, loc='right')

	if config['plotting']['colors']['colorbar']['Visible']:
		if config['plotting']['colors']['colorbar']['Location'] == 'inside':
			cb = fig.colorbar(contrail_heights_plotted, orientation = "horizontal", shrink=config['plotting']['colors']['colorbar']['ShrinkFactor'], aspect=22, 
							  pad=-.15, extendfrac=.02, format=config['plotting']['colors']['colorbar']['TickLabelFormat'])
			log.debug("Drawing colorbar turned ON")
			log.debug("Location = Inside plot")
			log.debug(f"Tick Label Format = '{config['plotting']['colors']['colorbar']['TickLabelFormat']}'")
		if config['plotting']['colors']['colorbar']['Location'] == 'bottom':
			cb = fig.colorbar(contrail_heights_plotted, orientation = "horizontal", shrink=config['plotting']['colors']['colorbar']['ShrinkFactor'], aspect=22, 
							  pad=.01, extendfrac=.02, format=config['plotting']['colors']['colorbar']['TickLabelFormat'])
			log.debug("Drawing colorbar turned ON")
			log.debug("Location = Bottom of plot")
			log.debug(f"Tick Label Format = '{config['plotting']['colors']['colorbar']['TickLabelFormat']}'")
		if config['plotting']['colors']['colorbar']['Location'] == 'right':
			cb = fig.colorbar(contrail_heights_plotted, orientation = "vertical", shrink=config['plotting']['colors']['colorbar']['ShrinkFactor'], aspect=22, 
							  pad=.01, extendfrac=.02, format=config['plotting']['colors']['colorbar']['TickLabelFormat'])
			log.debug("Drawing colorbar turned ON")
			log.debug("Location = Left of plot")
			log.debug(f"Tick Label Format = '{config['plotting']['colors']['colorbar']['TickLabelFormat']}'")

		if config['plotting']['colors']['colorbar']['LabelVisible']:
			cb.set_label(config['plotting']['colors']['colorbar']['Label'])
			log.debug("Colorbar label turned ON")
			log.debug(f"Label = '{config['plotting']['colors']['colorbar']['Label']}'")
	
	# Add logo to corner
	cwd = os.getcwd()
	ancillary_dir = os.path.join(cwd, config['misc']['AncillaryDirPath'])
	logo_sized = f"logo{config['plotting']['logo']['SizePix']}.png"
	logo = os.path.join(ancillary_dir, logo_sized)
	img = Image.open(logo)
	# ax.imshow(img, extent=(0.4, 0.6, 5, 5), zorder=15, alpha=cfg.DEF_LOGO_ALPHA)

	if region_id == 'US':
		proj_padding = 50
	else:
		proj_padding = 0
	fig.figimage(img, xo=ax.bbox.xmin + config['plotting']['logo']['MarginX'], yo=ax.bbox.ymin + (config['plotting']['logo']['MarginY'] + proj_padding), zorder=15, alpha=config['plotting']['logo']['Alpha'])

	region_name = region_name.replace(" ", "").upper()
	valid_time_str = valid_time.strftime('%Y%m%d_%H%M%S%Z')
	
	file_name = config['file']['NamingScheme'].format(**locals())

	out_dir = os.path.join(save_dir, region_id)
	
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	dest_path = os.path.join(out_dir, file_name + ".png")

	try:
		plt.savefig(dest_path, bbox_inches="tight", dpi=config['plotting']['DPI'])
	except:
		raise RuntimeError("Could not save file")
	else:
		log.info(f"Sucessfully saved plot file to {dest_path}")
		plt.close()

	tot_time = pd.Timestamp.now(tz='UTC') - now
	log.info(f"Done with plotting routine for region {region_name}, took {tot_time.total_seconds()} sec.")

def _dynamic_region_selector(sel_id, region_cfg):
	regions = region_cfg['regions']

	sel_region = {}

	for region in regions:
		if region['id'] == sel_id:
			sel_region = regions[regions.index(region)]

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

