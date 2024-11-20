import os
import tomllib
import datetime
import logging
from collections import namedtuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import from_levels_and_colors, to_rgba, ListedColormap
import matplotlib.patheffects as PathEffects
import cartopy
import cartopy.crs as crs
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.feature import NaturalEarthFeature
from PIL import Image

from __internal_funcs import plot_towns, draw_logo

PATH_TO_LOGCFG = "config/logging.conf"

with open(os.path.join(os.getcwd(), PATH_TO_LOGCFG), 'rb') as log_cfg:
	logging.config.dictConfig(tomllib.load(log_cfg))

global log
log = logging.getLogger("main.plot")

mpl.use('agg')

def plot_region(save_dir: str,
                product_info: dict,
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

	log.debug(f"Selected region {Region.name}")

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
		# if config['plotting']['units']['UseCustomRange']:
		levels = config['plotting']['units']['ftLevels']
		vmin = config['plotting']['units']['ftMinLevel']
		vmax = config['plotting']['units']['ftMaxLevel']

	if config['plotting']['colors']['UseCustomColormap']:
		palette = ListedColormap(tuple(config['plotting']['colors']['CustomColormapColors']))
	else:
		palette = config['plotting']['colors']['Colormap']

	
	contrail_heights_plotted = ax.contourf(long, lat, 
	                                       contrail_heights, 
										   levels = levels, 
										   cmap = palette, 
										   extend = "both", 
										   vmin = vmin, 
										   vmax = vmax, 
										   alpha = 1, 
										   transform=ccrs.PlateCarree(),
										   transform_first = True,
										   zorder=1)
	
	# Add lat/lon grids
	gridlines = ax.gridlines(linewidth=.8, 
	                         linestyle='--', 
							 color='gray', 
							 alpha=.5, 
							 draw_labels=True, 
							 x_inline=False, 
							 y_inline=False, 
							 zorder=11)

	gridlines.xlabel_style = {'size': 7}
	gridlines.ylabel_style = {'size': 7}
	gridlines.xlines = True
	gridlines.ylines = True
	gridlines.xlabels_top = False 
	gridlines.ylabels_right = False 

	xmin, xmax = ax.get_xlim()
	ymin, ymax = ax.get_ylim()
	xy = (xmin,ymin)
	width = xmax - xmin
	height = ymax - ymin

	# create the patch and place it in the back of countourf (zorder!)
	hatch_none = patches.Rectangle(xy, 
			                       width, 
			                       height, 
			                       hatch='x', 
			                       fill=None, 
			                       zorder=-10)
	ax.add_patch(hatch_none)

	xy_legend = (10.27, 1.5)#(1.01,0)
	xy_legendlab = (23, 6)

	hatch_legend = patches.Rectangle(xy_legend, 
	                                 width=0.25, 
	                                 height=0.25, 
	                                 hatch="x",
	                                 edgecolor="black",
	                                 facecolor="white",
	                                 clip_on=False,
	                                 transform=fig.dpi_scale_trans,#ax.transAxes,
	                                 zorder=35)
	ax.add_patch(hatch_legend)

	ax.annotate('No Contrails',
	            xy_legend,
	            xy_legendlab,
	            xycoords=fig.dpi_scale_trans,#'axes fraction',
	            textcoords='offset points',
	            annotation_clip=False,
	            transform=fig.dpi_scale_trans,#ax.transAxes,
	            fontsize=9,
	            zorder=35)

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

	ttl_init_str = init_time.strftime("%H:00 UTC %b %d %Y")
	fcasthr = str(fcst_hr)
	ttl_valid_str = valid_time.strftime("%H:00 UTC %b %d %Y")
	# plot_title = r"$\bf{"+"GFS\ Minimum\ Contrail\ Heights\ (ft\ &\ m\ MSL)"+"}$"+"\nInit: "+ttl_init_str+"   Forecast Hr: ["+fcasthr+"]   Valid: "+ttl_valid_str
	plot_title = f'{product_info["model"].upper()} {product_info["longname"]}\n'
	plot_desc = f'Initalized at {ttl_init_str}, Valid for {ttl_valid_str} (Forecast Hour {fcasthr})'
	ax.set_title(plot_title, 
	             loc='center', 
	             fontsize = 16, 
	             fontweight='bold')
	ax.set_title(plot_desc, 
	             loc='left', 
	             fontsize=11)
	ax.set_title(config['plotting']['Branding'], 
	             color='gray', 
	             fontweight='bold', 
				 fontsize=11, 
				 loc='right')

	if config['plotting']['colors']['colorbar']['Visible']:
		if config['plotting']['colors']['colorbar']['Location'] == 'inside':
			cb = fig.colorbar(contrail_heights_plotted, orientation = "horizontal", shrink=config['plotting']['colors']['colorbar']['ShrinkFactor'], aspect=22, 
							  pad=-.15, extendfrac=.05, format=config['plotting']['colors']['colorbar']['TickLabelFormat'])
			log.debug("Drawing colorbar turned ON")
			log.debug("Location = Inside plot")
			log.debug(f"Tick Label Format = '{config['plotting']['colors']['colorbar']['TickLabelFormat']}'")
		if config['plotting']['colors']['colorbar']['Location'] == 'bottom':
			cb = fig.colorbar(contrail_heights_plotted, orientation = "horizontal", shrink=config['plotting']['colors']['colorbar']['ShrinkFactor'], aspect=22, 
							  pad=.01, extendfrac=.05, format=config['plotting']['colors']['colorbar']['TickLabelFormat'])
			log.debug("Drawing colorbar turned ON")
			log.debug("Location = Bottom of plot")
			log.debug(f"Tick Label Format = '{config['plotting']['colors']['colorbar']['TickLabelFormat']}'")
		if config['plotting']['colors']['colorbar']['Location'] == 'right':
			cb = fig.colorbar(contrail_heights_plotted, orientation = "vertical", shrink=config['plotting']['colors']['colorbar']['ShrinkFactor'], aspect=22, 
							  pad=.01, extendfrac=.05, format=config['plotting']['colors']['colorbar']['TickLabelFormat'])
			log.debug("Drawing colorbar turned ON")
			log.debug("Location = Left of plot")
			log.debug(f"Tick Label Format = '{config['plotting']['colors']['colorbar']['TickLabelFormat']}'")

		if config['plotting']['colors']['colorbar']['LabelVisible']:
			cb.set_label(config['plotting']['colors']['colorbar']['Label'])
			log.debug("Colorbar label turned ON")
			log.debug(f"Label = '{config['plotting']['colors']['colorbar']['Label']}'")

	draw_logo(ax, f"logo{config['plotting']['logo']['SizePix']}.png", scale=5)
	plot_towns(ax, extent[0:2], extent[2:], scale_rank=config['plotting']['geography']['CityRankingFactor'], zorder=15)

	region_name = region_name.replace(" ", "").upper()
	valid_time_str = valid_time.strftime('%Y%m%d_%H%M%S%Z')

	model_name = product_info['model'].upper()
	product = product_info['name']
	
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

	log.debug(f"Selecting region")
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

	log.debug(f"Region selected")

	return Region(id=reg_id, name=reg_name, extent=reg_bbox, proj=sel_proj)

