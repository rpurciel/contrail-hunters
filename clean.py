import os
import shutil
from datetime import datetime
import sys
import glob
import logging

import pandas as pd

import config as cfg

log = logging.getLogger("clean")
log_format = logging.Formatter("[%(asctime)s:%(filename)s:%(lineno)s]%(levelname)s:%(message)s",
                               datefmt="%Y-%m-%d %H:%M:%S %Z",)
if cfg.DEBUG_LOG_TO_STDOUT:
	log_channel = logging.StreamHandler(stream=sys.stdout)

log_channel.setFormatter(log_format)
log.addHandler(log_channel)
log.setLevel(eval(cfg.DEBUG_LOGGING_LEVEL))

def clean_files_past_min_time():

	start_cleaner = datetime.now()

	cwd = os.getcwd()

	sys.path.insert(0, cwd)

	now = pd.Timestamp.now(tz='UTC')
	min_time = now - pd.Timedelta(hours=cfg.DEF_KEEP_FILES_HOURS)
	log.info(f"Plots valid before the following time will be deleted: {min_time}")

	dirs = sorted(glob.glob(cwd + "/" + cfg.DEF_PLOT_DIR_NAME + "/*/*", recursive=True))
	log.debug(f"Found directories: {dirs}")
	log.debug("Recursively checking...")

	for plt_dir in dirs:
		log.debug(f"Selected dir {plt_dir}")
		split_path = os.path.split(plt_dir)
		dir_datestr = os.path.basename(split_path[0])

		dir_yr = int(dir_datestr[:4])
		dir_m = int(dir_datestr[4:6])
		dir_d = int(dir_datestr[6:])
		dir_hr = int(split_path[1])

		dir_dt = pd.Timestamp(datetime(dir_yr, dir_m, dir_d, dir_hr), tz='UTC')
		log.debug(f"Dir time parsed as {dir_dt}")

		if dir_dt < min_time:
			log.debug("Directory time is before oldest allowed time.")
			log.debug(f"{min_time} <-- More Recent")
			log.debug(f"{dir_dt}")
			log.debug("Deleting...")

			try:
				shutil.rmtree(plt_dir)
			except Exception as e:
				log.error(f"Could not delete directory. Reason: {e}.")
			else:
				log.debug(f"REMOVAL --> {plt_dir}")
				log.info(f"Directory {plt_dir} and plots contained within has been removed.")
		else:
			log.debug("Directory time is current.")
			log.debug(f"{min_time}")
			log.debug(f"{dir_dt}        <-- More Recent")
			log.info(f"Directory {plt_dir} is current, will not be removed.")

	log.debug("Finished cleaning of old files.")
	return (datetime.now() - start_cleaner).total_seconds()





