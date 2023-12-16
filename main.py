import os
import tomllib
import logging
import logging.config as logcfg

import pandas as pd

from processor import ContrailProcessor

VERSION = "23.12.00_1"
#(year).(month).(version in year)_(dev version #)

PATH_TO_CONFIG = "config/config.conf"
PATH_TO_LOGCFG = "config/logging.conf"

with open(os.path.join(os.getcwd(), PATH_TO_LOGCFG), 'rb') as log_cfg:
	logcfg.dictConfig(tomllib.load(log_cfg))

global log
log = logging.getLogger("main")

if __name__ == "__main__":

	#ipython compatability
	__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

	with open(PATH_TO_CONFIG, 'rb') as config_file:
		config_opts = tomllib.load(config_file)

	now = pd.Timestamp.now(tz='UTC')

	log.info(f"CONTRAIL HUNTERS VERSION {VERSION}")
	log.info(f"Starting up....")
	log.info(f"Launched at {now}")

	proc = ContrailProcessor(config_opts)

	time = pd.Timestamp(year=2023, month=6, day=8, hour=6, tz='UTC')

	proc.populate_keys()

	# proc.load_files_from_dir('default')

	proc.aws_download_multithread()

	proc.plot_multiprocess()

	proc.delete_data_files()

	tot_time = pd.Timestamp.now(tz='UTC') - now

	log.info("CONTRAIL HUNTERS: ALL ROUTINES FINISHED")
	log.info(f"Total time: {tot_time}")
	log.info("Shutting down...")
