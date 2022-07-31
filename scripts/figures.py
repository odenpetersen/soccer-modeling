import pandas as pd
import numpy as np
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

def table_1(events):
    df = events[['eventName','subEventName']]
    df = df.groupby(['eventName','subEventName']).size()
    print(df)
    #df.unstack().plot(kind='bar');plt.show()

def main():
    events = pd.read_csv('../data/parsed_England.csv')
    matches = pd.read_json('../data/figshare/matches_England.json')
    #figure_1(matches)
    figure_2(events)
    #table_1(events)

if __name__=='__main__':
    main()
