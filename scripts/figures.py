import pandas as pd
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from collections import Counter

def figure_1(matches):
    goals = matches['label'].apply(lambda s : s.split(' ')).apply(lambda l : (int(l[-3]),int(l[-1])))
    counts = Counter(goals)
    dims = max(goals.apply(lambda t : max(t)))+1
    im = np.zeros((dims,dims))
    for i in range(dims):
        for j in range(dims):
            im[i][j] = counts[(i,j)]+counts[(j,i)]
    plt.imshow(im,cmap='hot')
    plt.show()

def figure_2(events):
    df = events.copy()
    df['timediff'] = df.eventSec.diff()
    same_match = df.matchId.shift(1) == df.matchId
    same_half = df.matchPeriod.shift(1) == df.matchPeriod
    df = df[same_match & same_half]
    event_types = set(df.eventName[~pd.isna(df.subEventName)])
    fig,ax = plt.subplots(3,3)
    print(event_types)
    for i,event in enumerate(event_types):
        data = df.timediff[df.eventName == event].apply(np.log10)
        data = data[data.apply(np.isfinite)]
        print(i%3,int(i/3))
        ax[i%3,int(i/3)].hist(data,label=f"Waiting time prior to {event}")
        ax[i%3,int(i/3)].legend()
    plt.show()

def figure_3(events):
    df = events.copy()
    df['next goal'] = df.index.to_series().apply(lambda x : min(df[df['tags'].apply(lambda l : 'Goal' in l) & (df.index > x)].index))
    df['next goal'] = df['next goal'].apply(lambda x : df[x].to_dict('records'))
    print(df.matchId)
    print(df['next goal'])
    same_match = df['next goal']['matchId'] == df.matchId
    same_half = df['next goal']['matchPeriod'] == df.matchPeriod
    df = df[same_match & same_half]
    df['timediff'] = df['next goal'] - df.eventSec
    plt.hist(df['timediff'])
    plt.show()

def figure_4(events):
    series = events['positions'].apply(lambda s : s.split(',')).apply(lambda t : (t[0][1:],t[1][:-1]))
    x = series.apply(lambda t : t[0]).apply(float)
    y = series.apply(lambda t : t[1]).apply(float)
    mask = ~((x==0)&(y==0))
    x = x[mask]
    y = y[mask]
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()

def figure_5(events):
    series = events['positions'].apply(lambda s : s.split(',')).apply(lambda t : (t[0][1:],t[1][:-1]))
    x = series.apply(lambda t : t[0]).apply(float)
    y = series.apply(lambda t : t[1]).apply(float)
    xdelta = x.diff()
    ydelta = y.diff()
    
    same_match = events.matchId.shift(1) == events.matchId
    same_half = events.matchPeriod.shift(1) == events.matchPeriod

    mask = ~((x==0)&(y==0)) & ~((xdelta==0)&(ydelta==0)) & same_match & same_half
    xdelta = xdelta[mask]
    ydelta = ydelta[mask]

    print(stats.shapiro(list(xdelta)[:5000]))
    print(stats.shapiro(list(ydelta)[:5000]))

    heatmap, xedges, yedges = np.histogram2d(xdelta, ydelta, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()

def table_1(events):
    df = events[['eventName','subEventName']]
    df = df.groupby(['eventName','subEventName']).size()
    print(df)
    #df.unstack().plot(kind='bar');plt.show()

def main():
    events = pd.read_csv('../data/parsed_England.csv')
    #matches = pd.read_json('../data/figshare/matches_England.json')
    #figure_1(matches)
    #figure_2(events)
    #figure_3(events)
    #figure_4(events)
    figure_5(events)
    #table_1(events)

if __name__=='__main__':
    main()
