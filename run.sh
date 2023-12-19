#!/bin/bash
# Logs everything
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>~/logs/contrail-hunters.log 2>&1

# Run using:
# bash -i ~/perlan_data/scripts/GFS_wrapper.sh

# Run GFS upper air winds plotting routine
echo 'Starting CONTRAIL HUNTERS...'
cd contrail-hunters
conda run --no-capture-output -n contrail-hunters-linux python3 main.py

# echo ' '
# echo 'Sending to WxDash...'
# scp ~/perlan_data/gfs_upper_air/animations/*.gif wwwwxdash@wxdash.com:graphics
# # scp ~/perlan_data/gfs_cross_sections/animations/*.gif wwwwxdash@wxdash.com:graphics
# # scp ~/perlan_data/gfs_soundings/animations/*.gif wwwwxdash@wxdash.com:graphics
echo 'Done.'