import sys
import shutil
import os
import glob
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

cwd = os.getcwd()
sys.path.insert(0, cwd)

import pandas as pd
import boto3

# ########## FOR TESTING ONLY, REMOVE ON PACKAGING ################
# sys.path.insert(0, '/Users/ryanpurciel/Development/contrail-hunters') 
# sys.path.insert(0, '/Users/rpurciel/Development/contrail-hunters')
# sys.path.insert(0, '/Users/ryanpurciel/Development/wexlib/src') 
# sys.path.insert(0, '/Users/rpurciel/Development/wexlib/src')
# ########## FOR TESTING ONLY, REMOVE ON PACKAGING ################

import download_gfs as gfs
import processor as cth
import config as cfg
import util as util
import clean as clean

def download_one_file(bucket: str, output: str, client: boto3.client, s3_file: str):
    """
    Download a single file from S3
    Args:
        bucket (str): S3 bucket where images are hosted
        output (str): Dir to store the images
        client (boto3.client): S3 client
        s3_file (str): S3 object name
    """
    client.download_file(
        Bucket=bucket, Key=s3_file, Filename=os.path.join(output, s3_file)
    )


files_to_download = ["file_1", "file_2", ..., "file_n"]
# Creating only one session and one client
session = boto3.Session()
client = session.client("s3")
# The client is shared between threads
func = partial(download_one_file, AWS_BUCKET, OUTPUT_DIR, client)

# List for storing possible failed downloads to retry later
failed_downloads = []

if __name__ == "__main__":

	try: #check to see if cfgrib is installed, because this is not checked for elsewhere
		import cfgrib
	except:
		raise ImportError("cfgrib not detected, must be installed to open files")

	start_all_tasks = datetime.now()

	log = logging.getLogger("main")
	log_format = logging.Formatter("[%(asctime)s:%(filename)s:%(lineno)s]%(levelname)s:%(message)s",
	                               datefmt="%Y-%m-%d %H:%M:%S %Z",)
	if cfg.DEBUG_LOG_TO_STDOUT:
		log_channel = logging.StreamHandler(stream=sys.stdout)

	log_channel.setFormatter(log_format)
	log.addHandler(log_channel)
	log.setLevel(eval(cfg.DEBUG_LOGGING_LEVEL))

	now = pd.Timestamp.now(tz='UTC')
	start_time = (now - pd.Timedelta(hours=cfg.DEF_MODEL_OUTPUT_LAG_HOURS)).floor(freq="6H")
	end_time = start_time + pd.Timedelta(hours=cfg.DEF_FORECAST_HOURS)
	data_period = pd.period_range(start=start_time, end=end_time, freq='3H')
	log.info(f"Downloading data between {start_time} - {end_time}")

	year = start_time.year
	month = start_time.month
	day = start_time.day
	hour = start_time.hour

	date_dir_str = f"{year}{str(month).zfill(2)}{str(day).zfill(2)}"

	cwd = os.getcwd()

	data_dir = os.path.join(cwd, cfg.DEF_DATA_DIR_NAME, date_dir_str, str(hour).zfill(2))
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

	plot_dir = os.path.join(cwd, cfg.DEF_PLOT_DIR_NAME, date_dir_str, str(hour).zfill(2))
	if not os.path.exists(plot_dir):
		try:
			os.makedirs(plot_dir)
		except Exception as e:
			log.error(f"Exception while creating data directory: {e}. Falling back to default ({cfg.DEF_FALLBACK_PLOT_DIR})")
		else:
			log.info(f"Plots loading into {plot_dir}")
	else:
		log.info(f"Plots loading into {plot_dir}")

	urls = []

	log.info("Listing files from AWS")
	for time in data_period:
		hours_from_start = round((time.to_timestamp(freq="H") - start_time.tz_localize(tz=None)).total_seconds() / 3600)
		log.info(f"File params: y={year} m={month} d={day} hr={hour} fcst={hours_from_start}")
		urls += [f"noaa-gfs-bdp-pds/gfs.{str(year).zfill(4)}{str(month).zfill(2)}{str(day).zfill(2)}/{str(hour).zfill(2)}/atmos/gfs.t{str(hour).zfill(2)}z.pgrb2.0p25.f{str(hours_from_start).zfill(3)}"]
		
	with tqdm.tqdm(desc="Downloading images from S3", total=len(files_to_download)) as pbar:
	    with ThreadPoolExecutor(max_workers=32) as executor:
	        # Using a dict for preserving the downloaded file for each future, to store it as a failure if we need that
	        futures = {
	            executor.submit(func, file_to_download): file_to_download for file_to_download in files_to_download
	        }
	        for future in as_completed(futures):
	            if future.exception():
	                failed_downloads.append(futures[future])
	            pbar.update(1)
	if len(failed_downloads) > 0:
	    print("Some downloads have failed. Saving ids to csv")
	    with open(
	        os.path.join(OUTPUT_DIR, "failed_downloads.csv"), "w", newline=""
	    ) as csvfile:
	        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
	        wr.writerow(failed_downloads)


	download_time = (datetime.now() - start_downloader).total_seconds() / 60
	log.info(f"Done downloading {len(data_period)} files")
	log.info(f"Took {download_time} minutes")
	# log.info(f"Starting plotting routine")
	# start_processor = datetime.now()
	# aws = s3fs.S3FileSystem(anon=True)
	# for url in urls:
	# 	start_file = datetime.now()
	# 	log.info(f"File url {url}")
	# 	try:
	# 		result, calc_time = cth.calculate_contrail_heights(data_file)
	# 	except Exception as e:
	# 		log.fatal(f"Contrail heights could not be calculated for {url}. Reason:")
	# 	else:
	# 		log.info(f"Success, took {calc_time} seconds")

	# 	for region in cfg.DEF_REGIONS_TO_PLOT:
	# 		log.info(f"Plotting region {region} of file {url}")
	# 		try:
	# 			plot_time = cth.plot_region(plot_dir, result, region)
	# 		except Exception as e:
	# 			log.fatal(F"Critical plotting error: {e}. Region {region} not plotted")
	# 		else:
	# 			log.info(f"Success, took {plot_time / 60} minutes")

	# 	file_time = (datetime.now() - start_file).total_seconds() / 60
	# 	log.info(f"Success for {url}, took {file_time} minutes")

	# processor_time = (datetime.now() - start_processor).total_seconds() / 60
	# log.info(f"Finished processing routines, took {processor_time} minutes")

	# log.info("Starting cleaning routine")
	# log.info("Deleting old plots...")
	# clean_time = clean.clean_files_past_min_time()

	# # if cfg.DEF_DELETE_DATA_AFTER_USE and not cfg.DEBUG_USE_STATIC_FILES:
	# # 	log.info("Deleting data files after use...")
	# # 	try:
	# # 		shutil.rmtree(os.path.join(cwd, cfg.DEF_DATA_DIR_NAME))
	# # 		log.info("Data removed sucessfully.")
	# # 	except Exception as e:
	# # 		log.error(f"Could not delete used data. Reason: {e}.")
	# # 		log.warning("Used data not removed. Watch storage space.")
	# # else:
	# # 	log.info("Removal of used data is turned off.")
	# # 	log.warning("Not removing used data. Watch storage space.")


	# all_tasks_time = (datetime.now() - start_all_tasks).total_seconds() / 60
	# log.info(f"Finished all routines for this time period")
	# log.info(f"Took {all_tasks_time} minutes")
	# log_channel.close()


