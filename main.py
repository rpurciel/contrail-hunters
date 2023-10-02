from ContrailProcessor import (download_current_gfs, calculate_contrail_heights,
							  plot_region)
from modeldata import ModelRunSeries, SingleRunData

if __name__ == "__main__":

	start_time = dt.datetime.now()
	print('\n====================================================================')
	print ("Script started at ", start_time)
	print('====================================================================\n')
	gfs_current = download_current_gfs(48, 3, "/Users/forecaster/Documents/GitHub/contrail-hunters/test")


	