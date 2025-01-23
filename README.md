# aggregate-reduce

This is a simple script that combines apartment data and then caps their peak as a percentage of the groups total. This is done to investigate the impact of combining the electricity loads of several households into one group and how coordination impacts the power effeciency efforts.

## Requirements

Besides the packages listed in the requirements the code assumes that there is data available with the columns being entities of energy demand and that the rows are a series of measurements. In the case study, one year of hourly data was used but the code does not require the data to be in chronological order so long as the measurements along a row are simultaneous. 
