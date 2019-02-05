"""
The point of this side project is to create a map of LA according to parking tickets received from last year (2018)
and to produce some interesting visualizations.
"""
# Imports
import numpy as np
import pandas as pd
import folium as fm # For the map, will only work in Jupyter since it needs a browser
from folium.plugins import FastMarkerCluster # In order to add map markers
from pyproj import Proj, transform
import matplotlib.pyplot as plt
import time

np.set_printoptions(threshold=np.inf, linewidth = 500, suppress=True)
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

"""
trimData
About: The csv that we're trying to load here is much too large to bring into Python in one shot (about 9-million x 
19). Another thing is that I'd like to throw away all entries that are from before 2018 - the data isn't in perfect 
chronological order though so I can't just look for where 2018 begins and throw everything else out. This function will 
grab all of the 2018/2019 incidents and save them in a new .csv.
Input: The name of the csv that contains the full data
Output: A csv containing only data from 2018/2019
"""
def trimData(file_in, file_out, batchsize):
    # This is going to be the new DataFrame that will contain only the rows that we want. We'll use the header from the
    # original dataframe
    d = pd.read_csv(file_in, delimiter = ',', header = 0, nrows = 0, dtype = str)
    dat_recent = pd.DataFrame(data = d)

    # The columns that we won't be using can be dropped
    dat_recent = dat_recent.drop(['Ticket number', 'Meter Id', 'Marked Time', 'RP State Plate', 'Plate Expiry Date',
                                'VIN', 'Make', 'Route', 'Agency', 'Violation code'], axis=1)

    i = 0
    not_done = True
    while not_done:

        print('processing batch', i, ', samples processed: ', i * batchsize)

        # load in batches of 1-million entries for processing per pass
        dat = pd.read_csv(file_in, delimiter = ',', header = 0, nrows = batchsize, skiprows = range(1, i*batchsize),
                          dtype = str)

        # Drop the columns that we don't need to save on space
        dat = dat.drop(['Ticket number', 'Meter Id', 'Marked Time', 'RP State Plate', 'Plate Expiry Date', 'VIN',
                         'Make', 'Route', 'Agency', 'Violation code'], axis = 1)

        # if the batch has less than 1-million entries then we know that this is the last pass
        i+=1
        if len(dat) < batchsize:
            not_done = False

        # replace the emply fields with "0000", I chose that so that checking for the year can be done by checking one
        # condition rather than two
        dat = dat.replace(np.nan, '0000')

        remove = [] # a list containing the index values to remove

        # Using the .at method in a for loop is 20x quicker than using iterrows
        for idx in dat.index:
            if dat.at[idx, 'Issue Date'][3] != '8':
                remove.append(idx)

        dat = dat.drop(dat.index[remove])
        dat_recent = dat_recent.append(dat)

    print(dat_recent)
    dat_recent.to_csv(file_out)
    print('done, new .csv saved as', file_out)

"""
cleanData
About: Once I have a .csv containing only the dates of interest I'd like to break up the date which is stored in a 
single cell as text into three cells saved as int values, along with getting rid of some other columns that I don't 
need. I'd also like to change the original format that the time was saved as.
Input: The name of the csv that contains the 2018 data
Output: A csv containing the same data in a more usable format
"""
def cleanData(file_in, file_out, batchsize):

    i = 0
    not_done = True

    # This will contain the newly formatted year/month/day stuff in three columns rather than just the one
    cols1 = ['Year', 'Month', 'Day']
    cols2 = ['Year', 'Month', 'Day', 'Issue time', 'Body Style', 'Color', 'Location', 'Violation Description',
             'Fine amount',	'Latitude',	'Longitude']

    data_cleaned = pd.DataFrame(columns = cols2)


    while not_done:

        newcols = pd.DataFrame(columns=cols1)

        print('processing batch', i, ', samples processed: ', i*batchsize)

        # load in batches of 1-million entries for processing per pass
        dat = pd.read_csv(file_in, delimiter=',', header=0, nrows=batchsize, skiprows=range(1, i * batchsize),
                          dtype = object)

        # if the batch has less than 1-million entries then we know that this is the last pass
        i += 1
        if len(dat) < batchsize:
            not_done = False

        hold_dict = {}

        for idx in dat.index:
            hold_dict[idx] = [int(dat.at[idx,'Issue Date'][3]), int(dat.at[idx,'Issue Date'][5:7]), int(dat.at[idx,
                                                                                            'Issue Date'][8:10])]

            # Tack "Los Angeles" on to the location just in case we end up using that column to designate location
            # and there is another street address with the same name somewhere else
            dat.at[idx,'Location'] = dat.at[idx,'Location'] + ' Los Angeles'
            dat.at[idx, 'Issue time'] = np.floor(float(dat.at[idx, 'Issue time']) / 100)

        hold_df = pd.DataFrame.from_dict(columns = cols1, data = hold_dict, orient = 'index')
        newcols = newcols.append(hold_df, ignore_index = True)

        dat = dat.drop(['Unnamed: 0','Issue Date'], axis = 1)

        data_cleaned = data_cleaned.append(pd.concat([newcols, dat], axis=1, sort = False), ignore_index = True)

    print(data_cleaned)
    data_cleaned.to_csv(file_out)
    print('done, new .csv saved as', file_out)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# To keep track of how long it takes
start_time = time.time()

# No need to run either if the csv's are already made
batchsize = 100000
trim_data = False
clean_data = False

if trim_data == True:
    trimData('parking-citations.csv', 'parking-citations-2018-present.csv', batchsize)

if clean_data == True:
    cleanData('parking-citations-2018-present.csv', '2018-parking-citations-cleaned.csv', batchsize)

# Load in the cleaned data as its own dataframe to work with
print('Loading data...')
working_data = pd.read_csv('2018-parking-citations-cleaned.csv', delimiter=',', header=0, dtype = object)

# Rename the first column which is duplicated upon loading the csv (csv saves the index and loading it in adds an index)
working_data = working_data.rename(columns = {'Unnamed: 0' : 'Index'})
print('Data loaded:')

# Print the sample
# print(working_data.head()) ... commented to avoid clutter

# - - - - - - - - - - - - - - - - For the map: - - - - - - - - - - - - - - - -

# Take the coorinates as their own dataframe so it can be manipulated, no need to throw out the rows with bad
# coordinate data from the full data set since they may contain other useful info

cbatch = 40000 # Number of previous incidents to plot, anything over this will bog down the map
coords = (working_data.loc[(len(working_data) - cbatch):, 'Latitude':'Longitude']).astype(float)

# Remove the cols with the 99999 values, if they're in one col they're in the other, no need to search both
coords = coords[coords['Latitude'] != 99999.0]

# coords are in x/y and we want lat/long, this is from the pyproj documentation
pm = '+proj=lcc +lat_1=34.03333333333333 +lat_2=35.46666666666667 +lat_0=33.5 +lon_0=-118 +x_0=2000000 ' \
     '+y_0=500000.0000000002 +ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs'

# convert to lat/long
x_in,y_in = coords['Latitude'].values, coords['Longitude'].values
lat,long = transform(Proj(pm, preserve_units = True), Proj("+init=epsg:4326"), x_in,y_in)

LA_coords = [34.05 , -118.24]
m = fm.Map(location=LA_coords, zoom_start=10.5)

# add map markers, plots as "long/lat" rather than "lat/long"
FastMarkerCluster(data=list(zip(long, lat))).add_to(m)
# display(m) # <- uncomment in jupyter


# - - - - - - - - - - - - - - - - For the figures - - - - - - - - - - - - - - - -
# Taking pieces of the data so I can manipulate them without affecting the original data, since I might want to use
# it for new things at some point

# Count the incidents per month
month_counts = working_data.groupby(by = 'Month', as_index=False).agg({'Index' : pd.Series.nunique})
month_counts = month_counts.astype(int)
month_counts = month_counts.sort_values(by = ['Month'], ascending = True)

f1 = plt.figure(figsize=(16, 7))
plt.bar(month_counts['Month'], month_counts['Index'])
plt.title('Violations Per Month (2018)')
plt.xlabel('Month')
plt.ylabel('Number of Violations')

# Violations according to hour
time_counts = working_data.groupby(by = 'Issue time', as_index=False).agg({'Index' : pd.Series.nunique})
time_counts = time_counts.astype(float)
time_counts = time_counts.sort_values(by = ['Issue time'], ascending = True)

f2 = plt.figure(figsize=(16, 7))
plt.bar(time_counts['Issue time'], time_counts['Index'])
plt.title('Violations Grouped by Hour of Occurrence (2018)')
plt.xlabel('Time (24 hr. clock)')
plt.ylabel('Number of Violations')

# Reasons for violations
reason_counts = working_data.groupby(by = 'Violation Description', as_index=False).agg({'Index' : pd.Series.nunique})
reason_counts = reason_counts.sort_values(by = ['Index'], ascending = False)
reason_counts = reason_counts[reason_counts['Index'] > 20000]

f3 = plt.figure(figsize=(16, 7))
plt.bar(reason_counts['Violation Description'], reason_counts['Index'])
plt.title('Top Violation Reasons')
plt.xlabel('Violation Reason')
plt.xticks(rotation=70)
plt.tight_layout()
plt.ylabel('Number of Violations')


plt.show()

print("--- %s seconds ---" % (time.time() - start_time))

