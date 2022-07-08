"""
Example of how to generate maps from CatalogResolver when the map size is large
enough to cause memory issues.

If you are making a 2D map from multiple stars, CatalogResolver will at some
point make a 4-dimensional array: x,y image dimensions, number of stars, and
number of cluster realizations (~300) for the uncertainty estimate.
If the image dimension is already large, the number of stars adds a factor of 10
and the cluster realizations adds a factor of 100, which can tank your RAM.

I have some handling in the code for large maps which uses numpy's "memmap"
capability and never tries to make the entire 4D array at once, looping
through the process instead.
There are 2 memmaps created:
    1) the cube of inverse distances from each star. This one can be reused for
        the exact same set of stars, and it will save some time.
    2) The result of all cluster realizations. If the result is a map, then this
         a series of 300 maps. The median along the 300 maps is the result and
         the 16 and 84 percentiles are the error bars.
         This can't be reused.

I attached an example of how to use this below; a lot of the work is still on
the user end, and I should probably build all this into the code at some point.

Created: July 8, 2022
"""
__author__ = "Ramsey Karim"



"""
The inv_dist_f() is the same.

Below that, we define the illumination_distance() function, and that is
different for the large map case.
"""

# Make distance array function (same as in small map case)
def inv_dist_f(coord):
    return 1./(4*np.pi * catalog.utils.distance_from_point_pixelgrid(coord, wcs_obj, distance_los)**2.)
    # Get inverse distance array AND INCLUDE 4PI (I think)
    # If I rewrote distance_from_point_, I could maybe do this with SkyCoord arrays.
    # As it's written now, this needs to be done as DataFrame.apply to each SkyCoord
    # inv_dist will be a 3D array

# extremely_large: True or False toggle
if not extremely_large:
    """
    Normal sized map
    """
    relevant_units = None
    extra_kwargs = {} # Empty dictionary
    # Make the entire inverse distance cube in one shot; not large, so no problem
    inv_dist = u.Quantity(list(catalog_df['SkyCoord'].apply(inv_dist_f).values))
    # Make fuv function to send to CatalogResolver.map_and_reduce_cluster
    def illumination_distance(fuv_flux_array):
        """
        :param fuv_flux_array: Quantity array in power units,
            should be the same shape as the inv_dist.shape[0]
        """
        return inv_dist * fuv_flux_array[:, np.newaxis, np.newaxis]
else:
    """
    Extremely large map
    """
    # The path can be anything, it doesn't matter
    memmap_fn = "/home/ramsey/Downloads/INVDIST_MEMMAP_oktodelete.dat"
    # Use this to store units, since memmap isn't Quantity
    relevant_units = {'inv_dist': None, 'flux': None}

    # catalog_df is the pandas DataFrame, the 'SkyCoord' column contains SkyCoords
    coords = list(catalog_df['SkyCoord'])
    if os.path.exists(memmap_fn):
        """
        If you have already saved a memmap of inverse distance for these stars
        (Saves some time, but not necessary)
        """
        print("Found existing memory mapped inverse distance grid, using that.")
        inv_dist = np.memmap(memmap_fn, dtype=np.float64, mode='r', shape=(len(catalog_df), *wcs_obj.array_shape))
        dummy_wcs = WCS(wcs_obj.to_header())
        dummy_wcs.array_shape = 2, 2
        relevant_units['inv_dist'] = (1./catalog.utils.distance_from_point_pixelgrid(coords[0], dummy_wcs, distance_los)**2.).unit
    else:
        """
        Make a new memmap of inverse distance
        """
        inv_dist = np.memmap(os.path.join(scoby.config.temp_dir, "INVDIST_MEMMAP_oktodelete.dat"), dtype=np.float64, mode='w+', shape=(len(catalog_df), *wcs_obj.array_shape))
        print(f"Solving inverse distance for {len(coords)} stars over a {wcs_obj.array_shape} grid:")
        from astropy.utils.console import ProgressBar # can put this at the top of code
        # Use ProgressBar to track inverse distance cube creation
        with ProgressBar(len(coords)-1) as bar:
            # Do all but the last star
            for i, c in enumerate(coords[:-1]):
                inv_dist[i, :] = inv_dist_f(c).to_value() # No units, just numbers
                bar.update()
        # Do last one manually and grab units
        x = inv_dist_f(coords[-1])
        inv_dist[i, :] = x.to_value()
        relevant_units['inv_dist'] = x.unit # Save units to that dictionary
        del x

    # Make a different version of illumination_distance()
    def illumination_distance(fuv_flux_array):
        """
        Same as above, but memory conscious.

        Note that this also reduces the result (result += chunk) while it loops,
        so the CatalogResolver doesn't need to sum across stars at the end.

        step_size = 3 means this does 3 stars at a time (slight speedup over
        just doing one star at a time).
        """
        result = np.zeros(wcs_obj.array_shape, dtype=np.float64)
        step_size = 3 # Do 3 stars at a time; slightly faster than just 1
        # Store the flux array units so we can put them back on later
        if relevant_units['flux'] is None:
            relevant_units['flux'] = fuv_flux_array.unit
        print(f"Summing gridded quantity \"{fuv_or_ionizing}\" over all stars:")
        for i in ProgressBar(range(0, len(catalog_df), step_size)):
            result += np.sum(inv_dist[i:i+step_size, :, :] * fuv_flux_array[i:i+step_size, np.newaxis, np.newaxis].to_value(), axis=0)
        return result

    extra_kwargs = {'reduce_func': False}

"""
At this point, we have a cube of inverse distances (a 2D image for each star)
and a function illumination_distance() that uses the flux array of each
cluster realization and the inverse distance cube to generate a flux map
realization.

We can pass the function illumination_distance() to CatalogResolver to get a
radiation field map (either FUV or ionizing).

We need to use the keyword extremely_large=True when we call the CatalogResolver
methods. CatalogResolver will make a second memory map to hold each cluster
realizations since that array can also be large.

Since illumination_distance() sums the radiation field for all stars while it
loops in order to save memory, we need to tell CatalogResolver not to sum at
the end. We do this by passing the argument reduce_func=False to the flux map
call. I did this here through the dictionary "extra_kwargs", but you can also
just do
>>> catr.get_FUV_flux(map_function, illumination_distance,
    extremely_large=True, reduce_func=False)
If you are doing a small map, the reduce_func keyword should not be given
    (i.e., don't just set it to True)

"""
# Get the median radiation field value and uncertainty
if fuv_or_ionizing == 'fuv': # G0
    val, uncertainty = catr.get_FUV_flux(map_function=illumination_distance, extremely_large=extremely_large, **extra_kwargs)
    quick_fix_units = lambda x: (x / radfield1d).decompose() # where radfield1d is the definition of 1 Habing unit using astropy units
elif fuv_or_ionizing == 'ionizing': # ionizing photon count
    # The ionizing photon flux call is exactly the same, just uses a different method name and units
    val, uncertainty = catr.get_ionizing_flux(map_function=illumination_distance, extremely_large=extremely_large, **extra_kwargs)
    quick_fix_units = lambda x: x.to(1/(u.cm**2 * u.s))

if extremely_large:
    # Fix units; the memmap array does not contain units, so we have to put them back on
    # We stored the units in the relevant_units dictionary
    val = u.Quantity(val, relevant_units['inv_dist']*relevant_units['flux'], copy=False)

# Delete the memmap that CatalogResolver created. This is the second memmap, it cannot be reused.
# The first memmap, for the inverse distances, can be reused for the exact same series of stars.
if extremely_large:
    # /home/ramsey/Downloads/ is my "temp_dir", which is set in config.py. Replace that with your temp_dir
    # You can set a temp_dir in config.py, but it will try to find your Downloads folder by default
    stresolver_memmap_filename = os.path.join(scoby.config.temp_dir, "STRESOLVER_MEMMAP_oktodelete.dat")
    if os.path.exists(stresolver_memmap_filename):
        os.remove(stresolver_memmap_filename)


# This is the result map! Median of all realizations
val = quick_fix_units(val) # express in useful units

# These are the uncertainties
lo, hi = uncertainty
lo = quick_fix_units(lo)
hi = quick_fix_units(hi)
