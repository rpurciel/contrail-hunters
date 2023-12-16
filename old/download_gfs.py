import sys
import os
import glob
from datetime import datetime
import ssl
import urllib.request
import warnings
import logging

import xarray as xr
import numpy as np
import pandas as pd

import config as cfg

warnings.filterwarnings("ignore")

#download params
DEF_DL_FORCE_FCST0 = False

log = logging.getLogger("dl-gfs")
log_format = logging.Formatter("[%(asctime)s:%(filename)s:%(lineno)s]%(levelname)s:%(message)s",
                               datefmt="%Y-%m-%d %H:%M:%S %Z",)
if cfg.DEBUG_LOG_TO_STDOUT:
    log_channel = logging.StreamHandler(stream=sys.stdout)

log_channel.setFormatter(log_format)
log.addHandler(log_channel)
log.setLevel(eval(cfg.DEBUG_LOGGING_LEVEL))

def download(save_dir, year, month, day, hour, **kwargs):
    """
    Downloads a single model file to a local directory. 

    Inputs: Directory to save model output files to, and year, month,
            day, and hour to download model file for.

    Returns: Success code, time (in seconds) for function to run,
             path to file
    """
    start_time = datetime.now()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log.debug(f"Selected time {str(year).zfill(4)}-{str(month).zfill(2)}-{str(day).zfill(2)} {str(hour).zfill(2)}:00:00 UTC")

    force_fcst0 = DEF_DL_FORCE_FCST0
    for arg, value in kwargs.items():
        if arg == 'force_fcst0':
            force_fcst0 = value

    if kwargs.get('forecast_hour'):
        forecast_hour = int(kwargs.get('forecast_hour'))
        log.debug(f"Downloading for forecast hour {forecast_hour}")

        url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs."+ \
            str(year)+str(month).zfill(2)+str(day).zfill(2)+ \
            "/"+str(hour).zfill(2)+"/atmos/gfs.t"+str(hour).zfill(2)+\
            "z.pgrb2.0p25.f"+str(forecast_hour).zfill(3)
        file_name = "gfs."+str(year)+ str(month).zfill(2) + str(day).zfill(2)+"."+str(hour).zfill(2)+"z.pgrb2.0p25.f"+ \
            str(forecast_hour).zfill(3)
    else:
        if force_fcst0:
            log.debug(f"INFO: Forcing forecast hour 0 to be downloaded")
            url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs."+ \
                str(year)+str(month).zfill(2)+str(day).zfill(2)+ \
                "/"+str(hour).zfill(2)+"/atmos/gfs.t"+str(hour).zfill(2)+\
                "z.pgrb2.0p25.anl"''
            file_name = "gfs."+str(year)+ str(month).zfill(2) + str(day).zfill(2)+"."+str(hour).zfill(2)+"z.pgrb2.0p25.f000"
        else:
            log.debug("Downloading analysis data")
            url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs."+ \
                str(year)+str(month).zfill(2)+str(day).zfill(2)+ \
                "/"+str(hour).zfill(2)+"/atmos/gfs.t"+str(hour).zfill(2)+\
                "z.pgrb2.0p25.anl"''
            file_name = "gfs."+str(year)+ str(month).zfill(2) + str(day).zfill(2)+"."+str(hour).zfill(2)+"z.pgrb2.0p25.anl"

    ssl._create_default_https_context = ssl._create_unverified_context

    dest_path = os.path.join(save_dir, file_name)

    log.debug("Starting downloader...")
    log.debug(f"URL: {url}")
    log.debug(f"Destination file name: {file_name}")
    try:
        urllib.request.urlretrieve(url, dest_path) #Retrieve the file and write it as a grbfile
    except urllib.error.URLError as e:
        log.fatal(f"Could not download file. Reason: {e.reason}")
        elapsed_time = datetime.now() - start_time
        return 0, elapsed_time.total_seconds(), e.reason
    else:
        log.debug("Finished downloading file")

    elapsed_time = datetime.now() - start_time
    return 1, elapsed_time.total_seconds(), dest_path





