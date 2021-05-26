# Darcy Driscoll 2020

# imports
import datetime as dt
# needed for pandas
import numpy as np 
# data analysis and manipulation
import pandas as pd 
# I use this to find a line of best fit for the data (overkill?)
from sklearn.linear_model import LinearRegression 
# graphs
import matplotlib.pyplot as plt 
# pretty graphs
import seaborn as sns 
plt.style.use('seaborn')

# import data
video_title = ""
retention = pd.read_csv('./data/{0}.csv'.format(video_title))
# drop Relative audience retention (%) because it is not used here
retention = retention.drop(columns='Relative audience retention (%)')
# rename these columns because they're easier to write
retention = retention.rename(columns={'Video position (%)' : 'video_position',
                                      'Absolute audience retention (%)' : 'absolute_retention'})                             
                                      
# Get raw rounded second marks in video
# This is required to ensure the x-ticks on the graph match the length of the video
video_length = 0  # in seconds -- can this be automated by hooking into the YouTube API?
diff = video_length / 99  # YouTube Studio displays 100 timestamps
seconds = [0]
for i in range(1, 100):
    seconds.append(seconds[i-1] + diff)

# round all second marks afterwards
seconds = [round(x) for x in seconds]

# convert seconds to time objects (i.e. video timestamps)
timestamps = []
for mark in seconds:
    minute = int(mark / 60)
    second = mark % 60
    timestamps.append(dt.time(0, minute, second))

# initialise plot
fig = plt.figure(figsize=(18,6))
ax = fig.add_subplot(111)

# plot line of best fit through the data so we see the general trend
# There is likely a better function to do this, LinearRegression is probably a bit hacky
lr = LinearRegression()
x = np.array(seconds)  # x-axis
y = retention['absolute_retention']  # y-axis
model = lr.fit(x.reshape(-1,1), y)  # fit to all data points
b0 = model.intercept_  # get y-intercept
b1 = model.coef_[0]  # get coefficients
x_range = [x.min(), x.max()]                      # get the bounds for x
y_range = [b0+b1*x_range[0], b0+b1*x_range[1]]    # get the bounds for y
plt.plot(x_range, y_range, c='blue', alpha=0.25)  # plot line using bounds

# in the retention dataframe, highlight gradients higher than the average and those hovering around 0
# (worse than the line of best fit = red; better than it = green; around zero = green)
colour_map = {'worse' : 'red', 'better' : 'green', 'stable' : 'green'}
colours = [colour_map['worse']]
# thresholds for stable and great diffs
stable_diff = 0.02
great_diff = 0.05
last = [None, None]
current = [None, None]
for i, row in enumerate(retention.itertuples()):
    # is this the first iteration of this loop?
    if last == [None, None]:
        # record this position and its retention
        last = [row.video_position, row.absolute_retention]
    else:
        current = [row.video_position, row.absolute_retention]
        # get the gradient between the previous position and this position
        grad = (current[1] - last[1]) / (current[0] - last[0])
        # compare gradient against thresholds
        if (grad > 0 and grad < stable_diff) or (grad < 0 and grad > -stable_diff):
            # around 0
            colours.append(colour_map['stable'])
        elif grad > b1 + great_diff:
            # greater than line of best fit
            colours.append(colour_map['better'])
        else:
            # worse than line of best fit
            colours.append(colour_map['worse'])
        last = current

# replace video positions with timestamps
retention['video_position'] = timestamps

# plot retention
for i in range(1, 100):
    # get two points to plot a line between
    rows = retention.iloc[i-1:i+1]
    # plot
    plt.plot(rows['video_position'], rows['absolute_retention'], colours[i])

# axes
ax.set_xlabel('Video timestamp', size=16)
ax.set_ylabel('Absolute audience retention (%)', size=16)
ax.set_title('Absolute Audience Retention Over Time\n{0}'.format(video_title), size=16)
# ticks
x = video_length / 12  # 12 ticks + 0 mark
np.arange(0, video_length + x, x).round()
ax.set_xticks(np.arange(0, video_length + x, x).round())

# show plot
plt.show()
# save plot
fig.savefig('graph.png')

# insert the colours as a column into retention
# could probably do this earlier for the 'plot retention' part
retention.insert(len(retention.columns), 'colour', colours)

# write highlighted timestamp ranges to file
with open('timestamps.txt', 'w') as f:
    # title
    f.write('\'{0}\' highlighted timestamp ranges'.format(video_title))
    # go through the dataframe and write in highlighted ranges
    last = None
    for row in retention.itertuples():
        if last is not None:
            if row.colour == colour_map['worse']:
                # ending highlight - write to file
                timestamp = '\n\n{0} {1} {2}'.format(last.strftime('%M:%S'), ' - ', row.video_position.strftime('%M:%S'))
                f.write('{0}\n\n----------------'.format(timestamp))
                last = None
        else:
            # starting new highlight
            if row.colour == colour_map['better'] or row.colour == colour_map['stable']:
                last = row.video_position