import sys
import os
import glob
from datetime import datetime
import logging

import pandas as pd

########## FOR TESTING ONLY, REMOVE ON PACKAGING ################
sys.path.insert(0, '/Users/ryanpurciel/Development/contrail-hunters') 
sys.path.insert(0, '/Users/rpurciel/Development/contrail-hunters')
sys.path.insert(0, '/Users/ryanpurciel/Development/wexlib/src') 
sys.path.insert(0, '/Users/rpurciel/Development/wexlib/src')
########## FOR TESTING ONLY, REMOVE ON PACKAGING ################

import wexlib.model.gfs as gfs
import wexlib.util.internal as util
import src.processor as cth
import src.config as cfg

if __name__ == "__main__":

	start_all_tasks = datetime.now()

	log = logging.getLogger("main")
	log_format = logging.Formatter("[%(name)s:%(asctime)s:%(filename)s:%(lineno)s:%(levelname)s] %(message)s",
	                               datefmt="%Y-%m-%d %H:%M:%S %Z",)
	if cfg.DEBUG_LOG_TO_STDOUT:
		log_channel = logging.StreamHandler(stream=sys.stdout)

	log_channel.setFormatter(log_format)
	log.addHandler(log_channel)
	log.setLevel(eval(cfg.DEBUG_LOGGING_LEVEL))

	now = pd.Timestamp.now(tz='UTC')
	start_time = now.floor(freq='18H')
	end_time = start_time + pd.Timedelta(hours=cfg.DEF_FORECAST_HOURS)
	data_period = pd.period_range(start=start_time, end=end_time, freq='3H')
	log.info(f"Downloading data between {start_time} - {end_time}")

	year = start_time.year
	month = start_time.month
	day = start_time.day
	hour = start_time.hour

	date_dir_str = f"{year}{str(month).zfill(2)}{str(day).zfill(2)}"

	srcdir = os.getcwd()
	cwd = os.path.dirname(srcdir)
	print(cwd)

	data_dir = os.path.join(cwd, cfg.DEF_DATA_DIR_NAME, date_dir_str)
	if not os.path.exists(data_dir):
		try:
			os.makedirs(data_dir)
		except Exception as e:
			log.error(f"Exception while creating data directory: {e}. Falling back to default ({cfg.DEF_FALLBACK_DATA_DIR})")
		else:
			log.info(f"Data loading into {data_dir}")
	else:
		log.info(f"Data loading into {data_dir}")

	if util.clean_idx(data_dir):
		log.info(f"Old .idx files removed from data dir")

	plot_dir = os.path.join(cwd, cfg.DEF_PLOT_DIR_NAME, date_dir_str)
	if not os.path.exists(plot_dir):
		try:
			os.makedirs(plot_dir)
		except Exception as e:
			log.error(f"Exception while creating data directory: {e}. Falling back to default ({cfg.DEF_FALLBACK_PLOT_DIR})")
		else:
			log.info(f"Plots loading into {plot_dir}")
	else:
		log.info(f"Plots loading into {plot_dir}")

	files = []

	start_downloader = datetime.now()
	if cfg.DEBUG_USE_STATIC_FILES == True:
		static_dir = os.path.join(cwd, cfg.DEBUG_STATIC_FILE_PATH)
		log.debug("Using static data files")
		log.debug(f"From dir: {static_dir}")
		if util.clean_idx(static_dir):
			log.debug(f"Old .idx files removed from static dir")
		files = sorted(glob.glob(static_dir + "/*"))
	else:
		log.info("Downloading files")
		for time in data_period:
			hours_from_start = round((time.to_timestamp(freq="H") - start_time.tz_localize(tz=None)).total_seconds() / 3600)

			log.info(f"File params: y={year} m={month} d={day} hr={hour} fcst={hours_from_start}")
			try:
				status, dl_time, file_path = gfs.download(data_dir, year, month, day, hour, 
				                                          forecast_hour=hours_from_start,
				                                          force_fcst0=True,)
			except Exception as e:
				log.error(f"Critical download error: {e}. File not downloaded.")
			else:
				files += [file_path]
				log.info(f"Success, took {dl_time} seconds")

	download_time = (datetime.now() - start_downloader).total_seconds()
	log.info(f"Done downloading {len(data_period)} files")
	log.info(f"Took {download_time} seconds")
	log.info(f"Starting plotting routine")
	start_processor = datetime.now()
	for file in files:
		start_file = datetime.now()
		log.info(f"File {file}")
		try:
			result, calc_time = cth.calculate_contrail_heights(file)
		except Exception as e:
			log.fatal(f"FATAL: Contrail heights could not be calculated for {file}. Reason:")
		else:
			log.info(f"Success, took {calc_time} seconds")

		for region in cfg.DEF_REGIONS_TO_PLOT:
			log.info(f"Plotting region {region} of file {file}")
			try:
				plot_time = cth.plot_region(plot_dir, result, region)
			except Exception as e:
				log.fatal(F"FATAL:Critical plotting error: {e}. Region {region} not plotted")
			else:
				log.info(f"Success, took {plot_time} seconds")

		file_time = (datetime.now() - start_file).total_seconds()
		log.info(f"Success for {file}, took {file_time} seconds")

	processor_time = (datetime.now() - start_processor).total_seconds()
	log.info(f"Finished processing routines, took {processor_time} seconds")

	all_tasks_time = (datetime.now() - start_all_tasks).total_seconds()
	log.info(f"Finished all routines for this time period")
	log.info(f"Took {all_tasks_time} seconds")
	log_channel.close()


