import os
from time import sleep
import tomllib
import datetime
import logging
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool, RLock, freeze_support
from threading import RLock as TRLock
from collections import namedtuple
import warnings
warnings.filterwarnings('ignore')

import s3fs
import boto3
import botocore
import botocore.config as botoconfig
from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map, thread_map
import pandas as pd
import xarray as xr
import numpy as np
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
import metpy
from metpy.units import units
from PIL import Image

def _plot_multiprocess_worker(save_dir: str,
							  logger,
							  config: dict,
							  file_and_pos: tuple):

	file = file_and_pos[0]
	pos = file_and_pos[1]

	regions_to_plot = config['plotting']['RegionsToPlot']

	file_name = os.path.basename(file)

	result = calculate_contrail_heights(file, logger)

	logger.info(result)

	prog = trange(len(regions_to_plot), position=pos, miniters=0, total=len(regions_to_plot)+2, leave=None)
	for region_pos in prog:

		sel_region = regions_to_plot[region_pos]
		prog.set_description(f"{file_name}: Plotting region '{sel_region}'", refresh=True)
		plot_region(save_dir, result, sel_region, logger, config)
		prog.update()

	prog.display(f"{file_name}: Finished plotting.", pos=pos)

def plot_region(save_dir: str,
				Calculation: namedtuple,
				region_id: str,
				log,
				config: dict):

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
	fig.figimage(img, xo=ax.bbox.xmin + config['plotting']['logo']['MarginX'], yo=ax.bbox.ymin + config['plotting']['logo']['MarginY'], zorder=15, alpha=config['plotting']['logo']['Alpha'])

	region_name = region_name.replace(" ", "").upper()
	valid_time_str = valid_time.strftime('%Y%m%d_%H%M%S%Z')
	
	file_name = config['file']['NamingScheme'].format(**locals())
	
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	dest_path = os.path.join(save_dir, file_name + ".png")

	try:
		plt.savefig(dest_path, bbox_inches="tight", dpi=config['plotting']['DPI'])
	except:
		raise RuntimeError("Could not save file")
	else:
		log.info(f"Sucessfully saved plot file to {dest_path}")
		plt.close()

	# buf = io.BytesIO()
	# plt.close()
	# buf.seek(0)
	# buf.close()

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


class ContrailProcessor:

	def __init__(self, config_opts):

		self.config = config_opts

		self.data_dir = self.config['download']['DataDirPath']
		self.plot_dir = self.config['plotting']['PlotDirPath']

		#Load in model parameters
		model_cfg_path = os.path.join(os.getcwd(), self.config['misc']['ModelConfigPath'])
		with open(model_cfg_path, 'rb') as model_cfg:
			self.model_opts = tomllib.load(model_cfg)

		self.sel_model = self.model_opts['models'][self.config['download']['Model']][0]

		self.bucket = self.sel_model['bucket_name']
		self.key_pattern = self.sel_model['key_pattern']

		#Load download interfaces
		self.boto3_session = boto3.Session()
		self.boto3_client = self.boto3_session.resource("s3", config=botoconfig.Config(signature_version=botocore.UNSIGNED))
		self.s3fs_client = s3fs.S3FileSystem(anon=True)

		self.log = self._start_logger()

	def _start_logger(self):

		log = logging.getLogger("main")
		log_format = logging.Formatter("[%(asctime)s:%(filename)s:%(lineno)s]%(levelname)s:%(message)s",
								   datefmt="%Y-%m-%d %H:%M:%S %Z",)

		if self.config['logging']['LogToSTDOUT']:
			log_channel = logging.StreamHandler(stream=sys.stdout)
		else:
			log_file_time = pd.Timestamp.now(tz='UTC').floor(freq="1min").strftime("%Y%m%d_%H%M%S")
			log_file_name = f"cth_{log_file_time}.log"
			log_file = os.path.join(self.config['logging']['LogDirPath'], log_file_name)
			log_channel = logging.FileHandler(log_file)

		log_channel.setFormatter(log_format)
		log.addHandler(log_channel)
		log.setLevel(eval(self.config['logging']['LoggingLevel']))

		log.info(f"Logging file created at {pd.Timestamp.now(tz='UTC')}")

		return log

	def populate_keys(self):

		now = pd.Timestamp.now(tz='UTC')

		start_time = (now - pd.Timedelta(hours=self.config['download']['ModelOutputLagHours'])).floor(freq="6H")
		end_time = start_time + pd.Timedelta(hours=self.config['download']['ForecastHours'])

		data_period = pd.period_range(start=start_time, end=end_time, freq='3H')
		self.log.debug(f"Downloading data between {start_time} - {end_time}")

		valid_keys = []
		valid_file_names = []

		year = str(start_time.year).zfill(4)
		month = str(start_time.month).zfill(2)
		day = str(start_time.day).zfill(2)
		hour = str(start_time.hour).zfill(2)

		for time in data_period:
			hours_from_start = round((time.to_timestamp(freq="H") - 
									  start_time.tz_localize(tz=None)).total_seconds() / 3600)

			fcst_hr = str(hours_from_start).zfill(3)
			product = ""

			self.log.debug(f"File params: y={year} m={month} d={day} hr={hour} fcst={hours_from_start}")

			key_head = self.key_pattern.format(**locals())
			if self.s3fs_client.exists(f"{self.bucket}/{key_head}"):
				valid_keys += [key_head]
				file_name = f"{self.sel_model['name']}{product}.{year}{month}{day}{hour}.f{fcst_hr}{self.sel_model['file_ext']}"
				valid_file_names += [file_name]
			else:
				self.log.info(f"No valid file found for time {year}-{month}-{day} {hour}Z, f{fcst_hr}, skipping...")

		self.keys = valid_keys
		self.file_names = valid_file_names

		return self.keys

	def aws_download_multithread(self):
		'''
		Thin wrapper for multithreaded downloading.
		'''
		self.data_files = []

		tqdm.set_lock(TRLock())
		try:
			with ThreadPoolExecutor(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as executor:
				executor.map(partial(self._aws_download_multithread_worker, self.data_dir), self.keys, self.file_names, range(1, len(self.keys)+1, 1))
		except Exception as e:
			print(e)

	def _aws_download_multithread_worker(self,
										 save_dir: str,
										 s3_file: str,
										 file_name: str, 
										 progress_pos: int):

		file_size = int(self.boto3_client.Object(self.bucket, s3_file).content_length)
		pretty_file_name = os.path.basename(s3_file)
		file_path = os.path.join(save_dir, file_name)

		self.data_files += [file_path]

		try:
			with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=pretty_file_name, ascii=" ▖▘▝▗▚▞█", total=file_size, leave=None, position=progress_pos) as progress:
				self.boto3_client.Bucket(self.bucket).download_file(s3_file, file_path, Callback=progress.update)
				progress.close()	
				progress.display(f"{pretty_file_name}: Finished downloading.", pos=progress_pos)
		except Exception as e:
			print(e)

	#Plotting routines

	def plot_multiprocess(self):
		
		tqdm.set_lock(RLock())
		p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
		p.map(partial(_plot_multiprocess_worker, self.plot_dir, self.log, self.config), zip(self.data_files, range(1, len(self.data_files)+1, 1)))

	
def calculate_contrail_heights(data_file, log):

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

	return Calculation(contrail_heights=hgt_contrail, lat=latitude, lon=longitude, init_time=init_time, valid_time=dt_obj, fcst_hr=fcst_hr)


