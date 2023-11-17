from ContrailProcessor import (download_current_gfs, calculate_contrail_heights,
							  plot_region)
from modeldata import ModelRunSeries, SingleRunData

from datetime import datetime

if __name__ == "__main__":

	start_time = datetime.now()
	print('\n====================================================================')
	print ("Script started at ", start_time)
	print('====================================================================\n')
	gfs_current = download_current_gfs(48, 3, "/Users/rpurciel/Development/contrail-hunters/test")

	print(gfs_current)

	