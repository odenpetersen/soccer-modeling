import numpy as np
import pandas as pd

#Training functions should train a model and return a lambda function wrapper that takes in feature vectors and outputs the relevant number

def train_next_goal_home_model(events):
    return lambda match,event : None

def train_next_goal_home_prior_model(events):
    return lambda match : None

def train_waiting_time_model(events):
    return lambda match,event : None

def train_waiting_time_prior_model(events):
    return lambda match : None


def train(events):
    #Given most recent event, predict which team will score the next goal
    next_goal_home_model = train_next_goal_home_model(events)

    #Given prior information about the match, and who controls the ball at kick-off, predict which team will score the first goal
    next_goal_home_prior_model = train_next_goal_home_prior_model(events)

    #Given the most recent event, predict the waiting time until the next goal
    waiting_time_model = train_waiting_time_model(events)

    #Given prior information about the match, and who controls the ball at kick-off, predict the waiting time until the first goal
    waiting_time_prior_model = train_waiting_time_prior_model(events)

    models = {'next_goal_team'      :   next_goal_home_model,
              'waiting_time'        :   waiting_time_model,
              'next_goal_team_prior':   next_goal_home_prior_model,
              'waiting_time_prior'  :   waiting_time_prior_model}

    return models

def simulate_recursive(models,lambda_H,lambda_A,time_left):
    W_H = np.random.exponential(1/lambda_H)
    W_A = np.random.exponential(1/lambda_A)

    if time_left >= 0:
        next_goal_vector = np.array([0,1]) if W_A<W_H else np.array([1,0])
        return next_goal_vector + simulate_recursive(models,lambda_H,lambda_A,time_left-min(W_H,W_A))
    else:
        return np.array([0,0])

def simulate(models,match,match_event,lambda_H,lambda_A,n=2):
    #Parameter estimation
    P_H_R = models['next_goal_home'](match,match_event)
    P_A_R = 1-P_H_R
    E_W_R = models['waiting_time'](match,match_event)
    lambda_H_R = P_H_R/E_W_R
    lambda_A_R = P_A_R/E_W_R

    #Recursive Monte Carlo a bunch of times
    simulations = []
    for _ in range(n):
        W_H = np.random.exponential(1/lambda_H_R)
        W_A = np.random.exponential(1/lambda_A_R)

        time_left = 45*60-(match_event['eventSec']%(45*60))
        time_left -= min(W_A,W_H)

        if time_left<0:
            simulations.append(np.array([0,0]))
            continue

        next_goal_vector = np.array([0,1]) if W_A<W_H else np.array([1,0])

        simulations.append(next_goal_vector + simulate_recursive(lambda_H,lambda_A,time_left))

    frequencies = Counter(simulations)

    return {k:v/n for k,v in frequencies.items()}

def main():
    events=pd.read_csv('../data/parsed_England.csv').sort_by(['matchId','eventSec'])
    #Feature vector for prior match info; should be a numpy array
    events['match'] = events.apply(lambda row : ,axis=1)
    #Feature vector for match events; should be a numpy array
    events['event'] = events.apply(lambda row : ,axis=1)

    #Goals scored prior to a particular match event
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
    models = train(events)

    match_prior_parameters = {matchId : (models['next_goal_home'](match), lambda_A) for matchId in set(events.matchId)}

    events['scores'] = events.apply(lambda row : simulate(models,row.match,row.event,lambda_H,lambda_A),axis=1)

    #For events in the first half, also simulate the second half
    prior_half_simulation = {matchId : simulate_recursive(match_prior_parameters[matchId][0],match_prior_parameters[matchId][1],time_left) for matchId in set(events['matchId'])}
    events['scores'] = events.apply(lambda row : {scoreFirst+scoreSecond : probFirst * probSecond for scoreFirst,probFirst in row.scores.items() for scoreSecond,probSecond in prior_half_simulation[row.matchId].items()},axis=1)

    #Finally, add previous goals
    events['scores'] = events.apply(lambda row : {row.prior_goals+k for k,v in row.scores})

    #Evaluate simulation quality
    events['surprisal'] =

if __name__=='__main__':
    main()
