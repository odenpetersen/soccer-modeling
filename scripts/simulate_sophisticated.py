import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
tqdm.pandas()

eventTypePredicates = {'Goal' : lambda row : 'Goal' in row.tags and row.eventName in ['Shot','Free Kick'],
                       'Duel' : lambda row : row.eventName == 'Duel',
                       'Pass' : lambda row : row.eventName == 'Pass',
                       'Home Team' : lambda row : row.teamsData[str(row.teamId)]['side']=='home',
                       'Away Team' : lambda row : row.teamsData[str(row.teamId)]['side']=='away',
                       'Yval Left' : lambda row : row.positions[0] < 105/3,
                       'Yval Mid' : lambda row : 105/3 <= row.positions[0] and row.positions[0] < 2*105/3,
                       'Yval Right' : lambda row : 2*105/3 <= row.positions[0],
                       'Xval Up' : lambda row : row.positions[1] < 68/3,
                       'Xval Mid' : lambda row : 68/3 <= row.positions[1] and row.positions[1] < 2*68/3,
                       'Xval Down' : lambda row : 2*68/3 <= row.positions[1]
                        }

#Model to produce a transition matrix give match info
def train_transition_matrix(startIds,endIds,eventTypeKeys):
    freqs = Counter(zip(startIds,endIds))
    #Uses Laplace smoothing to prevent transition probabilities from being zero
    matrix = [[freqs[(start,end)]+1 for end in eventTypeKeys] for start in eventTypeKeys]
    matrix = np.matrix(matrix)
    return matrix/np.sum(matrix,axis=1)

#Model to produce a waiting time estimate given event and match info
def train_waiting_time(states,times,eventTypeKeys):
    mean_waiting_time = [np.mean([t for s,t in zip(states,times) if s==state]) for state in eventTypeKeys]
    return mean_waiting_time

def simulate_recursive(eventType,lambdas,time_left,eventTypeKeys):
    samples = [np.random.exponential(1/param) for param in lambdas[eventType]]
    new_time_left = time_left - min(samples)
    new_state = np.argmin(samples)
    
    if new_time_left < 0:
        return np.array([0,0])
    
    marginal_goals = np.array([0,0])
    if eventTypeKeys[newState][0] == 'Goal':
        if eventTypeKeys[newState][1] == 'Home Team':
            marginal_goals = np.array([1,0])
        else:
            marginal_goals = np.array([0,1])

    return marginal_goals + simulate_recursive(new_state,lambdas,new_time_left,eventTypeKeys)

def convolve(d1,d2):
    d = {}
    for i in d1.keys():
        for j in d2.keys():
            if i+j in d:
                d[i+j] += d1[i]+d2[j]
            else:
                d[i+j] = d1[i]+d2[j]
    return d

def main():
    matches = pd.read_json('../data/figshare/matches_England.json')
    events = pd.read_csv('../data/parsed_England.csv')

    print("Evaluating positions")

    events['positions'] = events['positions'].apply(eval)

    print("Evaluating teamsData")

    events.teamsData = events.teamsData.progress_apply(eval)

    print("Generating event types")

    events['eventType'] = events.progress_apply(lambda row : tuple([k for k in eventTypePredicates.keys() if eventTypePredicates[k](row)]),axis=1)

    print("Generating unique event type keys")

    eventTypeKeys = {v:k for k,v in enumerate(set(events['eventType']))}

    print(eventTypeKeys)

    print("Filtering events")

    events['eventTypeId'] = events.eventType.progress_apply(eventTypeKeys.get)
    events = events[events['eventType'].apply(len)==4]

    print(events)

    same_match = events.matchId.shift(1) == events.matchId
    same_half = events.matchPeriod.shift(1) == events.matchPeriod

    startIds = events['eventTypeId'][same_match & same_half]
    endIds = events['eventTypeId'].shift(1)[same_match & same_half]
    timedeltas = events.eventSec.diff()[same_match & same_half]

    print(timedeltas)

    print("Training transition matrix")

    transition_matrix = train_transition_matrix(startIds, endIds, eventTypeKeys)

    print(transition_matrix)
    
    print("Training waiting times")

    waiting_times = train_waiting_time(startIds, timedeltas, eventTypeKeys)

    print(waiting_times)

    print("Generating lambdas")

    lambdas = [probs/waiting_times[i] for i,probs in enumerate(transition_matrix)]

    print(lambdas)

    print("Generating simulations")

    events['simulated'] = events.apply(lambda row : Counter([simulate_recursive(initial_state, lambdas, 45*60-(row.eventSec%(45*60)), eventTypeKeys) for _ in range(10)]),axis=1)

    #Simulate second half if needed
    prior_half_simulations = Counter([simulate_recursive(np.random.sample(),lambdas,45*60,eventTypeKeys) for _ in range(20)])
    #Convolve with events
    events['simulated'] = events.apply(lambda row : convolve(row.simulated, prior_half_simulations),axis=1)

    #Add prior goals
    matchId = None
    goals=0
    prior_goals = {}
    for i in events.index:
        if matchId is None or matchId != events['matchId'][i]:
            matchId = events['matchId']
            goals=0
        if events['tags'].apply(lambda l : 'Goal' in l)[i] and (events['eventName'][i] in ['Shot','Free Kick']):
            goals += 1
        prior_goals[i] = goals

    events['prior_goals'] = pd.Series(prior_goals)
    events['simulated'] = events.apply(lambda row : {k+row.prior_goals:v for k,v in row.simulated.items()})

    events['simulated'] = events['simulated'].apply(lambda d : {k:v/sum(d.values()) for k,v in d.items()})

    print(events['simulated'])
   
if __name__=='__main__':
    main()
