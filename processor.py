import os
import tarfile
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
from pathlib import Path
warnings.filterwarnings('ignore')

import s3fs
import boto3
import botocore
import botocore.config as botoconfig
from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map, thread_map
import pandas as pd
from paramiko import SSHClient
from scp import SCPClient

from plot import plot_region
import contrail_calc

PATH_TO_LOGCFG = "config/logging.conf"

with open(os.path.join(os.getcwd(), PATH_TO_LOGCFG), 'rb') as log_cfg:
	logcfg.dictConfig(tomllib.load(log_cfg))

global log
log = logging.getLogger("main")

def _plot_mp_worker(save_dir: str,
                    product_info: dict,
				    config: dict,
				    file_and_pos: tuple):

	log.debug("Spinning up minimum heights worker process...")

	file = file_and_pos[0]
	pos = file_and_pos[1]

	regions_to_plot = config['plotting']['RegionsToPlot']

	file_name = os.path.basename(file)

	sel_func = product_info['calcfunc']

	#dynamic selection of calculation function based on config
	calc_func = getattr(contrail_calc, sel_func)
	result = calc_func(file)

	failed_regions = []
	prog = trange(len(regions_to_plot), position=pos, miniters=0, total=len(regions_to_plot)+2, leave=None)
	for region_pos in prog:

		sel_region = regions_to_plot[region_pos]
		prog.set_description(f"{file_name}: Plotting region '{sel_region}'", refresh=True)
		try:
			plot_region(save_dir, product_info, result, sel_region, config)
		except Exception as e:
			log.fatal(f"Region {sel_region} not plotted, saving for rerun...")
			log.fatal(f"Reason: {e}")
			failed_regions += [sel_region]
		prog.update()

	prog.display(f"{file_name}: Finished plotting.", pos=pos)

	#Below code implements retries of failed plots. I don't think it needs to
	#be implemented ASAP, but it's here.

	# if failed_regions:
	# 	log.info(f"{len(failed_regions)} regions failed to plot, retrying...")
	# 	#retry on regions that failed the first time
	# 	numiters = 1
	# 	while failed_regions:
	# 		this_iter = failed_regions.copy()

	# 		for region in this_iter:
	# 			log.info(f"Retrying plotting of {region} (num tries: {numiters}) (max tries: {maxiters})")
	# 			try:
	# 				plot_region(save_dir, result, region, config)
	# 			except:
	# 				log.fatal(f"Could not plot region {region}, retrying or stopping...")
	# 			else:
	# 				log.info(f"{region} plotted successfully.")

	log.debug("Min heights worker process spinning down...")

class ContrailProcessor:

	def __init__(self, config_opts):

		self.config = config_opts

		self.data_dir = self.config['download']['DataDirPath']
		self.plot_dir = self.config['plotting']['PlotDirPath']
		self.archive_dir = self.config['misc']['ArchiveDirPath']

		if not os.path.exists(self.data_dir):
			os.makedirs(self.data_dir)

		if not os.path.exists(self.plot_dir):
			os.makedirs(self.plot_dir)

		#Load in model parameters
		model_cfg_path = os.path.join(os.getcwd(), self.config['misc']['ModelConfigPath'])
		with open(model_cfg_path, 'rb') as model_cfg:
			self.model_opts = tomllib.load(model_cfg)

		self.sel_model = self.model_opts['models'][self.config['download']['Model']][0]

		self.bucket = self.sel_model['bucket_name']
		self.key_pattern = self.sel_model['key_pattern']

		self.ssh_connection = False

		#Load download interfaces
		self.boto3_session = boto3.Session()
		self.boto3_client = self.boto3_session.resource("s3", config=botoconfig.Config(signature_version=botocore.UNSIGNED))
		self.s3fs_client = s3fs.S3FileSystem(anon=True)

		#Construct namedtuples of product info (for selection in calculating/plotting functions)
		product_info_path = os.path.join(os.getcwd(), self.config['misc']['ProductConfigPath'])
		with open(product_info_path, 'rb') as product_info:
			self.products = tomllib.load(product_info)


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

		data_period = pd.period_range(start=start_time, end=end_time, freq=f"{self.config['download']['ForecastTimeIncrements']}H")
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

	def clean_dir(self, id_or_path):

		if id_or_path == "data":
			path = self.data_dir
		elif id_or_path == "output":
			path = self.plot_dir
		elif id_or_path == "archive":
			path = self.archive_dir
		else:
			path = id_or_path
			if not os.path.exists(path):
				raise ValueError("Path is not valid")

		abs_dir = os.path.join(os.getcwd(), path)
		log.info(f"Deleting files from selected directory ({abs_dir})")

		if abs_dir[-1:] == "/":
			abs_dir = abs_dir[:-1]

		if Path(abs_dir).is_symlink():
			linked_dir = os.readlink(abs_dir) + "/"
		elif Path(abs_dir).is_dir():
			linked_dir = abs_dir + "/"
		else:
			raise ValueError("Path must be a directory")

		files = glob.glob(linked_dir + "*", recursive=True)
		log.info(f"Found {len(files)} tagged for deletion")

		for file in files:
			if file == linked_dir:
				pass
			else:
				if Path(file).is_file():
					log.debug(f"Removed file {file}")
					os.remove(file)
				else:
					log.debug(f"Removed directory {file} (and files/dirs within)")
					shutil.rmtree(file)

		log.info(f"All files in {abs_dir} deleted. ({len(files)})")

	def archive_run(self, tag=None, include_data=False):

		now = pd.Timestamp.now(tz='UTC')
		log.info(f"Archiving run at {now} UTC")

		time_str = now.strftime("%Y%m%d_%H%M%S")

		if tag:
			archive_name = f"{tag}_cthrun_{time_str}Z_archive.tar.gz"
		else:
			archive_name = f"cthrun_{time_str}Z_archive.tar.gz"

		archive_dir = self.config['misc']['ArchiveDirPath']
		if not os.path.exists(archive_dir):
			os.makedirs(archive_dir)

		archive_file = os.path.join(archive_dir, archive_name)

		with tarfile.open(archive_file, "w:gz") as tar:
			tar.add(self.plot_dir, arcname=(os.path.basename(self.plot_dir)))
			if include_data:
				tar.add(self.data_dir, arcname=(os.path.join("data", os.path.basename(self.data_dir))))

	def _init_ssh_connection(self):
		log.info("Initiating SSH connection to server")
		self.ssh = SSHClient()
		self.ssh.load_system_host_keys()
		self.ssh.connect(self.config['connection']['ServerName'], username=self.config['connection']['UserName'])
		log.info("Done")

		log.info("Initiating SFTP client through SSH connection")
		self.sftp = self.ssh.open_sftp()
		log.info("Done")

		self.connection_name = f"{self.config['connection']['UserName']}@{self.config['connection']['ServerName']}"

		return True

	def remove_old_files_from_server(self):

		if not self.ssh_connection:
			self.ssh_connection = self._init_ssh_connection()

		now = pd.Timestamp.now(tz='UTC')
		log.info(f"Removing old files from server at {now} UTC")
		log.info(f"Address = {self.connection_name}")

		files = self.sftp.listdir(self.config['connection']['RemoteDir'])

		for file in files:
			if file.endswith('.png'):
				file_path = os.path.join(remote_dir, file)
				self.sftp.remove(file_path)
				log.info(f"{self.connection_name}: Removed {file} at {file_path}")
			else:
				log.info(f"{self.connection_name}: File {file} in remote dir not removed")

	def send_files_to_server_sftp(self):

		if not self.ssh_connection:
			self.ssh_connection = self._init_ssh_connection()

		now = pd.Timestamp.now(tz='UTC')
		log.info(f"Transferring files to server at {now} UTC")
		log.info("Via: SFTP")
		log.info(f"Address = {self.connection_name}")

		abs_path = os.path.join(os.getcwd(), self.plot_dir)

		output_files = glob.glob(abs_path + "/**", recursive=True)

		files_to_transfer = []
		for file in output_files:
			if file == abs_path:
				pass
			else:
				if Path(file).is_file():
					files_to_transfer += [file]
				else:
					pass

		with tqdm(miniters=0, mtotal=len(files_to_transfer), desc=self.connection_name,  ascii=" >-", leave=None) as progress:
			for file in files_to_transfer:
				file_name = os.path.basename(file)
				remote_path = os.path.join(self.config['connection']['RemoteDir'], file_name)
				progress.set_description(desc=f'{self.connection_name}:{file_name}')
				try:
					self.sftp.put(file, remote_path)
				except Exception as e:
					print(e)
				progress.update()

			prog.display(f"{self.connection_name}: Sending all files finished.", pos=pos)

	def send_files_to_server_scp(self):

		now = pd.Timestamp.now(tz='UTC')
		log.info(f"Transferring files to server at {now} UTC")
		log.info("Via: SCP")
		log.info(f"Address = {self.connection_name}")

		abs_path = os.path.join(os.getcwd(), self.plot_dir)

		output_files = glob.glob(abs_path + "/**", recursive=True)

		files_to_transfer = []
		for file in output_files:
			if file == abs_path:
				pass
			else:
				if Path(file).is_file():
					files_to_transfer += [file]
				else:
					pass

		ssh = SSHClient()
		ssh.load_system_host_keys()
		ssh.connect(self.config['connection']['ServerName'], username=self.config['connection']['UserName'])
		scp = SCPClient(ssh.get_transport())

		connection_name = f"{self.config['connection']['UserName']}@{self.config['connection']['ServerName']}"

		with tqdm(miniters=0, mtotal=len(files_to_transfer), desc=connection_name,  ascii=" >-", leave=None) as progress:
			for file in files_to_transfer:
				file_name = os.path.basename(file)
				progress.set_description(desc=f'{connection_name}:{file_name}')
				try:
					scp.put(file, remote_path='graphics/')
				except Exception as e:
					print(e)
				progress.update()

			prog.display(f"{connection_name}: Sending all files finished.", pos=pos)

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

	def plot_mp(self, product_id: str):

		sel_product = self.products['products'][product_id][0]
		if not sel_product:
			raise NotImplementedError("Specified product was not found.")

		out_dir = os.path.join(self.plot_dir, sel_product['dirname'])
		
		tqdm.set_lock(RLock())
		p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
		p.map(partial(_plot_mp_worker, out_dir, sel_product, self.config), zip(self.data_files, range(1, len(self.data_files)+1, 1)))

	

