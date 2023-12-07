from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool, RLock, freeze_support
from random import random
from threading import RLock as TRLock
from time import sleep
import os

import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config
from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map, thread_map

from processor import calculate_contrail_heights, plot_region
import config as cfg
# import util as util
# import clean as clean

NUM_SUBITERS = 9

AWS_BUCKET = "noaa-gfs-bdp-pds"
DATA_DIR = "/Users/rpurciel/Documents/Test/Multithread Downloads/Data/"
OUTPUT_DIR = "/Users/rpurciel/Documents/Test/Multithread Downloads/Plots/"

def downloader(bucket: str, output: str, file_and_pos: (str, int)):

    session = boto3.Session()
    resource = session.resource("s3", region_name='us-east-1', config=Config(signature_version=UNSIGNED))

    s3_file = file_and_pos[0]
    progress_pos = file_and_pos[1]

    meta_data = resource.Object(bucket, s3_file).content_length
    total_length = int(meta_data)
    file_name = os.path.basename(s3_file)
    file_path = os.path.join(output, file_name)

    try:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=file_name, total=total_length, leave=None, position=progress_pos) as progress:
            resource.Bucket(bucket).download_file(s3_file, os.path.join(output, file_path), Callback=progress.update)
            progress.close()
            progress.display(f"Finished downloading file {file_name}, starting subroutines...")
            subprog = trange(len(cfg.DEF_REGIONS_TO_PLOT) + 2, position=progress_pos + 1, miniters=1, leave=None)
            for i in subprog:
                if i == 0:
                    subprog.set_description("    @ Calculating heights", refresh=True)
                    heights, _ = calculate_contrail_heights(file_path)
                    subprog.update(1)
                elif i == len(cfg.DEF_REGIONS_TO_PLOT) + 1:
                    subprog.set_description("    @ Deleting data file", refresh=True)
                    sleep(5)
                    subprog.update(1)
                else:
                    subprog.set_description(f"    @ Plotting region {i}", refresh=True)
                    plot_region(OUTPUT_DIR, heights, cfg.DEF_REGIONS_TO_PLOT[i-1])
                    subprog.update(1)
            subprog.close()
            progress.display(f"All tasks for file {file_name} finished.", pos=progress_pos)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    freeze_support()  # for Windows support

    files_to_download = ["gfs.20231117/12/atmos/gfs.t12z.pgrb2.0p25.f000",
                     "gfs.20231117/12/atmos/gfs.t12z.pgrb2.0p25.f003",
                     "gfs.20231117/12/atmos/gfs.t12z.pgrb2.0p25.f006",
                     "gfs.20231117/12/atmos/gfs.t12z.pgrb2.0p25.f009",
                     "gfs.20231117/12/atmos/gfs.t12z.pgrb2.0p25.f012",
                     "gfs.20231117/12/atmos/gfs.t12z.pgrb2.0p25.f015",
                     "gfs.20231117/12/atmos/gfs.t12z.pgrb2.0p25.f018",
                     "gfs.20231117/12/atmos/gfs.t12z.pgrb2.0p25.f021",
                     "gfs.20231117/12/atmos/gfs.t12z.pgrb2.0p25.f024"]

    print(type(zip(files_to_download, range(1, len(files_to_download)*2, 2))))

    tqdm.set_lock(RLock())
    p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    p.map(partial(downloader, AWS_BUCKET, DATA_DIR), zip(files_to_download, range(1, len(files_to_download)*2, 2)))

    # print("Multi-threading")
    # tqdm.set_lock(TRLock())
    # with ThreadPoolExecutor(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
    #     p.map(partial(downloader, AWS_BUCKET, DATA_DIR, client), files_to_download, range(1, len(files_to_download)*2, 2))
    # print("All tasks finished!")