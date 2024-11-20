import os
import tomllib
import logging
import logging.config as logcfg
import argparse

import pandas as pd

from processor import ContrailProcessor

VERSION = "02.24.11"
#(version in year).(year).(month)_(dev version #)

PATH_TO_CONFIG = "config/config.conf"
PATH_TO_LOGCFG = "config/logging.conf"

with open(os.path.join(os.getcwd(), PATH_TO_LOGCFG), 'rb') as log_cfg:
	logcfg.dictConfig(tomllib.load(log_cfg))

global log
log = logging.getLogger("main")

parser = argparse.ArgumentParser(description=f'Run CONTRAIL HUNTERS v. {VERSION}')
parser.add_argument('--load-static-data', 
                    help='load files from data directory instead of downloading', 
                    action='store_true', 
                    dest='static')
parser.add_argument('--local-only', 
                    help='run program locally, do not connect to destination server', 
                    action='store_true', 
                    dest='local')
parser.add_argument('--clean-data-before-run', 
                    help='remove old data at the beginning of processing', 
                    action='store_true', 
                    dest='clean_data')
parser.add_argument('--clean-plots-before-run', 
                    help='remove old plots at the beginning of processing', 
                    action='store_true', 
                    dest='clean_plots')
parser.add_argument('--keep-data-after-run', 
                    help='prevent deletion of data files from directory when done processing', 
                    action='store_true', 
                    dest='keep_data')
parser.add_argument('--time', 
                    help='specifiy a time (UTC) to start the forecast period from', 
                    nargs=4, 
                    default=argparse.SUPPRESS, 
                    metavar=('yyyy', 'mm', 'dd', 'hh'), 
                    type=int)

if __name__ == "__main__":

	#ipython compatability
	__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

	args = parser.parse_args()

	with open(PATH_TO_CONFIG, 'rb') as config_file:
		config_opts = tomllib.load(config_file)

	now = pd.Timestamp.now(tz='UTC')

	print(f"CONTRAIL HUNTERS v. {VERSION}")

	log.info(f"CONTRAIL HUNTERS v. {VERSION}")
	log.info(f"Starting up....")
	log.info(f"Launched at {now}")
	log.debug(f"Args passed (if any): {args}")

	proc = ContrailProcessor(config_opts)

	if args.clean_plots:
		log.info("Cleaning old plots")
		proc.clean_dir('output')

	if args.clean_data:
		log.info("Cleaning old data")
		proc.clean_dir('data')
		
	if 'time' in args:
		time = pd.Timestamp(year=int(args.time[0]), 
		                    month=int(args.time[1]), 
		                    day=int(args.time[2]), 
		                    hour=int(args.time[3]), 
		                    tz='UTC')
		proc.populate_keys(time)
	else:
		proc.populate_keys()

	if args.static:
		proc.load_files_from_dir('default')
		print("Loading files from static directory")
	else:
		proc.aws_download_multithread()

	proc.clean_idx_files()

	#Calculation/Plotting routines
	proc.plot_mp('MinHeightsNB')
	proc.plot_mp('MinHeightsLB')
	proc.plot_mp('MinHeightsHB')

	tot_time = pd.Timestamp.now(tz='UTC') - now

	log.info("CONTRAIL HUNTERS: ALL ROUTINES FINISHED")
	log.info(f"Total time: {tot_time}")
	log.info("Shutting down...")

	#End of run utils
	if not args.local:
		proc.remove_old_files_from_server()
		proc.send_files_to_server_sftp()

	proc.archive_run()

	if not args.keep_data and not args.static:
		proc.clean_dir('data')
		proc.clean_dir('output')

	log.info("END")

