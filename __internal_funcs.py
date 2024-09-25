import os

from numpy import arange
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cartopy.mpl.geoaxes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.feature import NaturalEarthFeature

def plot_towns(ax: cartopy.mpl.geoaxes.GeoAxes, 
               lon_bounds: tuple[float, float],
               lat_bounds: tuple[float, float], 
               resolution: str = '10m',
               scale_rank: int = 5, 
               transform: cartopy.crs = ccrs.PlateCarree(), 
               zorder: int = 3) -> None:
    """
    This function will download the 'populated_places' shapefile from
    NaturalEarth, trim the shapefile based on the limits of the provided
    lat & long coords, and then plot the locations and names of the towns
    on a given GeoAxes.

    Inputs
        - ax, cartopy.mpl.geoaxes.GeoAxes: 
                A GeoAxes object to plot onto.
        - lat_bounds, tuple: 
                A tuple of size 2 containing the latitude bounds of the plot area.
                Standard ordering is from lower to upper bound.
        - lon_bounds, tuple: 
                A tuple of size 2 containing the longitude bounds of the plot area.
                Standard ordering is from lower to upper bound.
        - resolution, str (optional, def '10m'):
                A string specifying what resolution of town data to download. Default
                is the finest resolution.
        - scale_rank, int (optional, def 5):
                Int representing the upper bound of the 'scale rank' of the towns to
                plot, from most-to-least culturally relevant. e.g. 1-2 will plot only
                the largest towns, >5 will plot tiny towns.
        - transform, cartopy.crs (optional, def ccrs.PlateCarree()):
                A transformation to use for the map. Should match transformation for
                GeoAxes object.
        - zorder, int (optional, def 3):
                Int specifying the zorder value to be fed into matplotlib.
    
    Returns
        None
    """

    from adjustText import adjust_text 

    shp = shpreader.Reader(shpreader.natural_earth(resolution=resolution, category='cultural', name='populated_places'))

    #get town names
    names = []
    x_pts = []
    y_pts = []

    for town in shp.records():
        if int(town.attributes['SCALERANK']) <= scale_rank:
            x_pts.append(town.attributes['LONGITUDE'])
            y_pts.append(town.attributes['LATITUDE'])
            name = town.attributes['NAME_EN']
            names.append(f'{name}')

    #create data frame and index by the region of the plot
    all_towns = pd.DataFrame({'names': names, 'x': x_pts, 'y': y_pts})
    region_towns = all_towns[(all_towns.y<max(lat_bounds)) & (all_towns.y>min(lat_bounds))
                           & (all_towns.x>min(lon_bounds)) & (all_towns.x<max(lon_bounds))]

    #plot the locations and labels of the towns in the region
    ax.scatter(region_towns.x.values, region_towns.y.values, 
               s=10,
               c ='black', 
               marker= '.', 
               transform=transform, 
               zorder=zorder)

    town_names = []
    for row in region_towns.itertuples():
        name = ax.text(float(row.x), 
                       float(row.y) * 0.9995, 
                       row.names,
                       fontsize=9, 
                       transform=transform,
                       style='italic',
                       horizontalalignment='left',
                       verticalalignment='top',
                       clip_box=ax.bbox)
        town_names.append(name)

    #use adjustText library to autoadjust town names to prevent ugly overlapping
    adjust_text(town_names)

def draw_logo(ax: mpl.axes.Axes | cartopy.mpl.geoaxes.GeoAxes, 
              logo_rel_path: str = 'logo.png',
              corner: str = 'lowerleft',
              scale: float = 1,
              zorder: int = 30) -> None:

    """
    This function will use a locally stored logo image, and 
    then plot the image on a corner of the plot.

    Inputs
        - ax, matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes: 
                An Axes or GeoAxes object to plot onto.
        - logo_rel_path, str (optional, def 'logo.png'):
                A string specifying what file to use as a logo. Files are stored
                in the 'logos' subdirectory.
        - corner, str (optional, def 'lowerleft')
                A string specifying what corner (of the internal bounding box) to
                place the logo at. Default is the lower left corner.
        - scale, int (optional, def 1):
                Optional scaling factor of the logo's size.
        - zorder, int (optional, def 30):
                Int specifying the zorder value to be fed into matplotlib.

    Returns
        None
    """

    mpl.use('agg')

    try:
        with open(f'ancillary/{logo_rel_path}', 'rb') as file:
            logo_img = plt.imread(file)
    except Exception as OpenExcept:
        raise OpenExcept

    logobox = OffsetImage(logo_img, zoom=0.05 * scale)
    logobox.image.axes = ax

    if 'lower' in corner:
        logo_vpos = 0
    elif 'upper' in corner:
        logo_vpos = 1
    else:
        logo_vpos = 0

    if 'left' in corner:
        logo_hpos = 0
    elif 'right' in corner:
        logo_hpos = 1
    else:
        logo_hpos = 0

    logo = AnnotationBbox(logobox, 
                         (logo_hpos, logo_vpos),
                         xybox=(10., 10.),
                         xycoords='axes fraction',
                         boxcoords="offset points",
                         box_alignment=(0, 0),
                         pad=0.0,
                         frameon=False,
                         zorder=zorder)

    ax.add_artist(logo)

def generate_verification_plot(
       lat: float | list[float, ...], 
       lon: float | list[float, ...], 
       save_dir: str,
       extent_factor: int = 5,
    ) -> None:
    """
    This function generates a plot of the input lat lon,
    to be used for verification of the input location.
    Plot is saved to [save_dir]/location_verification.png

    Inputs
        - lat, float: 
                Latitude of requested point. A list of latitudes is supported,
                if one is passed plot is centered around the last point.
        - lon, float: 
                Longitude of requested point. A list of latitudes is supported,
                if one is passed plot is centered around the last point.
        - save_dir, str: 
                The desired plot save destination.
        - extent_factor, int (optional: def 5):
                The factor (in degrees) to calculate the extent of the plot.
                e.g. 5 means 5 degrees out from the center on each side.

    Returns
        None
    """

    mpl.use('agg')

    #Generate figure and populate with features (borders, water, towns, etc.)
    fig = plt.figure(figsize=(22,16))
    ax = plt.axes(projection = ccrs.PlateCarree())

    if (type(lat) or type(lon)) is list:
        extent = [lon[len(lon)]-extent_factor,
                  lon[len(lon)]+extent_factor,
                  lat[len(lat)]-extent_factor,
                  lat[len(lat)]+extent_factor]
    else:
        extent = [lon-extent_factor,
                  lon+extent_factor,
                  lat-extent_factor,
                  lat+extent_factor]

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    states = NaturalEarthFeature(category="cultural", scale="50m",
                                                  facecolor="none",
                                                  name="admin_1_states_provinces")
    ax.add_feature(states, linewidth=1.0, edgecolor="black")
    ax.coastlines('50m', linewidth=1.5)
    ax.add_feature(cartopy.feature.LAKES.with_scale('10m'), linestyle='-', linewidth=0.5, alpha=1,edgecolor='blue',facecolor='none')
    ax.add_feature(cfeature.BORDERS, linewidth=1.5)
    ax.add_feature(cfeature.LAND)
    plot_towns(ax, extent[0:2], extent[2:])

    #Add latitude/longitude gridlines and labels
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=True)
    gl.right_labels = True
    gl.left_labels = True
    gl.bottom_labels = True
    gl.top_labels = True
    gl.xlocator = mticker.FixedLocator(arange(extent[0], extent[1], 2))
    gl.ylocator = mticker.FixedLocator(arange(extent[2], extent[3], 2))

    #Plot requested point(s) and save
    if (type(lat) or type(lon)) is list:
        plt.plot(lon, lat, marker='x', color='red', transform=ccrs.PlateCarree(), zorder=15)
        plt.annotate(f"Beginning\n{lat[len(lat)]}, {lon[len(lon)]}", 
                     (lon[len(lon)], lat[len(lat)]-0.3), 
                     color='red', 
                     fontweight='bold', 
                     horizontalalignment='center', 
                     verticalalignment='top',
                     transform=ccrs.PlateCarree(), 
                     zorder=15)
    else:
        plt.plot(lon, lat, marker='x', color='red', transform=ccrs.PlateCarree(), zorder=15)
        plt.annotate(f"Selected Point\n{lat}, {lon}", 
                     (lon, lat-0.3), 
                     color='red', 
                     fontweight='bold', 
                     horizontalalignment='center', 
                     verticalalignment='top',
                     transform=ccrs.PlateCarree(), 
                     zorder=15)

    draw_logo(ax)
    plt.savefig(os.path.join(save_dir, "location_verification.png"), bbox_inches="tight", dpi=200)
    
    plt.close()

def remove_idx_files(target_directory: str) -> None:

    import os

    abs_dir = os.path.join(os.getcwd(), self.data_dir)

    log.info("Deleting used .idx files from data dir")

    files = glob.glob(abs_dir + "/*.idx")
    if not files:
        log.info("No IDX files found")
    else:
        for file in files:
            os.remove(file)


