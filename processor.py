import os
from time import sleep
import tomllib
import datetime
import logging
import logging.config as logcfg
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool, RLock, freeze_support
from threading import RLock as TRLock
from collections import namedtuple
import warnings
import glob
import shutil
warnings.filterwarnings('ignore')

import s3fs
import boto3
import botocore
import botocore.config as botoconfig
from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map, thread_map
import pandas as pd

from plot import plot_region
from contrail_calc import calculate_contrail_heights

PATH_TO_LOGCFG = "config/logging.conf"

with open(os.path.join(os.getcwd(), PATH_TO_LOGCFG), 'rb') as log_cfg:
	logcfg.dictConfig(tomllib.load(log_cfg))

global log
log = logging.getLogger("main")

def _plot_multiprocess_worker(save_dir: str,
							  config: dict,
							  file_and_pos: tuple):

	log.debug("Spinning up plotting/calculating worker process...")

	file = file_and_pos[0]
	pos = file_and_pos[1]

	regions_to_plot = config['plotting']['RegionsToPlot']

	file_name = os.path.basename(file)

	result = calculate_contrail_heights(file)

	prog = trange(len(regions_to_plot), position=pos, miniters=0, total=len(regions_to_plot)+2, leave=None)
	for region_pos in prog:

		sel_region = regions_to_plot[region_pos]
		prog.set_description(f"{file_name}: Plotting region '{sel_region}'", refresh=True)
		plot_region(save_dir, result, sel_region, config)
		prog.update()

	prog.display(f"{file_name}: Finished plotting.", pos=pos)

	log.debug("Worker process spinning down...")

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

		# global log
		# log = logging.getLogger(__name__)
		# log_format = logging.Formatter("[%(asctime)s:%(filename)s:%(lineno)s]%(levelname)s:%(message)s",
		# 						   datefmt="%Y-%m-%d %H:%M:%S %Z",)

		# log_file_time = pd.Timestamp.now(tz='UTC').strftime("%Y%m%d_%H%M%S%f")
		# log_file_name = f"cth_{log_file_time}.log"
		# self.log_file = os.path.join(self.config['logging']['LogDirPath'], log_file_name)

		# if self.config['logging']['LogToSTDOUT']:
		# 	log_channel = logging.StreamHandler(stream=sys.stdout)
		# else:
		# 	log_channel = logging.FileHandler(self.log_file)

		# log_channel.setFormatter(log_format)
		# log.addHandler(log_channel)
		# log.setLevel(eval(self.config['logging']['LoggingLevel']))

		# log.info(f"Logging file created at {pd.Timestamp.now(tz='UTC')}")

		# print(self.log_file)

	def populate_keys(self, *args):

		now = pd.Timestamp.now(tz='UTC')
		'''
		Populate this processor object with valid AWS keys
		for downloading of model data.

		Parameters
		----------
		bucket  : str
			Bucket name to use.
		key_pattern  : str
			Key pattern string to use. If glob_match is False, must be a
			path to a folder containing files (not subfolders), otherwise
			nothing will be downloaded.
			If glob_match is True, uses standard terminology.
		glob_match  : bool, optional [default: False]
			Turns on glob-style matching for key names.

		Returns
		-------
		out  : A list of valid file keys for the given bucket.
		'''
		modelLagHours = self.config['download']['ModelOutputLagHours']

		if args:
			now = args[0]
			modelLagHours = 0
			if len(args) > 1:
				fcst_len = args[1]

		start_time = (now - pd.Timedelta(hours=modelLagHours)).floor(freq="6H")
		end_time = start_time + pd.Timedelta(hours=self.config['download']['ForecastHours'])

		data_period = pd.period_range(start=start_time, end=end_time, freq=f'{self.config['download']['ForecastTimeIncrements']}H')
		log.debug(f"Downloading data between {start_time} - {end_time}")

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

			log.debug(f"File params: y={year} m={month} d={day} hr={hour} fcst={hours_from_start}")

			key_head = self.key_pattern.format(**locals())
			if self.s3fs_client.exists(f"{self.bucket}/{key_head}"):
				valid_keys += [key_head]
				file_name = f"{self.sel_model['name']}{product}.{year}{month}{day}{hour}.f{fcst_hr}{self.sel_model['file_ext']}"
				valid_file_names += [file_name]
			else:
				log.info(f"No valid file found for time {year}-{month}-{day} {hour}Z, f{fcst_hr}, skipping...")

		self.keys = valid_keys
		self.file_names = valid_file_names

		return self.keys

	def load_files_from_dir(self, path):

		if path == 'default':
			path = self.data_dir

		if path[-1:] != "/":
			path = path + "/"

		abs_path = os.path.join(os.getcwd(), self.data_dir)

		self.data_files = glob.glob(path + "*" + self.sel_model['file_ext'])

		log.info(f"Loaded {len(self.data_files)} files from path {abs_path}")

	def clean_idx_files(self):

		abs_dir = os.path.join(os.getcwd(), self.data_dir)

		log.info("Deleting used .idx files from data dir")

		files = glob.glob(abs_dir + "/*.idx")
		if not files:
			log.info("No IDX files found")
		else:
			for file in files:
				os.remove(file)

	def delete_data_files(self):

		abs_dir = os.path.join(os.getcwd(), self.data_dir)
		log.info(f"Deleting data files from defined data directory ({abs_dir})")
		shutil.rmtree(abs_dir)
		os.makedirs(abs_dir)
		log.info("All files deleted.")

	def delete_plots(self):

		abs_dir = os.path.join(os.getcwd(), self.plot_dir)
		log.info(f"Deleting plots from defined plot directory ({abs_dir})")
		shutil.rmtree(abs_dir)
		os.makedirs(abs_dir)
		log.info("All files deleted.")

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

		log.debug("Spinning up download thread...")

		file_size = int(self.boto3_client.Object(self.bucket, s3_file).content_length)
		pretty_file_name = os.path.basename(s3_file)
		file_path = os.path.join(save_dir, file_name)

		self.data_files += [file_path]

		log.info(f"Starting download: {pretty_file_name}")

		try:
			with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=pretty_file_name, ascii=" ▖▘▝▗▚▞█", total=file_size, leave=None, position=progress_pos) as progress:
				self.boto3_client.Bucket(self.bucket).download_file(s3_file, file_path, Callback=progress.update)
				progress.close()	
				progress.display(f"{pretty_file_name}: Finished downloading.", pos=progress_pos)
				log.info(f"Download finished: {pretty_file_name}")
				log.debug("Download thread spinning down...")
		except Exception as e:
			log.fatal(e)



	#Plotting routines

	def plot_multiprocess(self):
		
		tqdm.set_lock(RLock())
		p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
		p.map(partial(_plot_multiprocess_worker, self.plot_dir, self.config), zip(self.data_files, range(1, len(self.data_files)+1, 1)))

	

