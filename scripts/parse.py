import pandas as pd
from dateutil import parser

data = "../data/figshare/"
data2 = "../data"
competitions = ["England","Germany","World_Cup","France","Spain","Italy","European_Championship"]

def parse():
    ignored_columns = []

    df = pd.concat([pd.read_json(data + f"events_{competition}.json") for competition in competitions])

    print("Data loaded")

    ignored_columns += ['eventId','subEventId']

    names = pd.read_csv(data+"tags2name.csv")
    names.index = names['Tag']
    df['tags'] = df['tags'].apply(lambda ts : [names['Description'][t['id']] for t in ts])

    players = pd.read_json(data+"players.json")
    df = df.merge(players,how='left',left_on='playerId',right_on='wyId').drop('wyId',axis=1)

    df['player'] = df['playerId'].apply(str) + " - " + df['firstName'] + " " + df['middleName'] + " " + df['lastName']
    
    ignored_columns += ['playerId','firstName','middleName','lastName']
    
    matches = pd.concat([pd.read_json(data+f"matches_{competition}.json") for competition in competitions])
    df = df.merge(matches,how='left',left_on='matchId',right_on='wyId').drop('wyId',axis=1)

    df['teamData'] = df.apply(lambda r : r.teamsData[str(r.teamId)],axis=1)
    df['positions'] = df['positions'].apply(lambda t : t[0])
    df['positions'] = df.apply(lambda r : (r.positions['x'],r.positions['y']) if r.teamData['side']=='home' else (100-r.positions['x'],100-r.positions['y']) if r.teamData['side']=='away' else None,axis=1)
    df['positions'] = df['positions'].apply(lambda t : (105*t[0]/100,68*t[1]/100))

    ignored_columns += ['teamData']

    #df['date'] = df['date'].apply(parser.parse)
    #df['birthDate'] = df['birthDate'].apply(lambda x : None if pd.isna(x) else parser.parse(x))
    #df['age'] = df['date']-df['birthDate']

    ignored_columns += ['birthDate']

    df = df.drop(ignored_columns,axis=1)
    df = df.sort_values(['matchId','Pulling via Rebase
The --rebase option can be used to ensure a linear history by preventing unnecessary merge commits. Many developers prefer rebasing over merging, since it’s like saying, "I want to put my changes on top of what everybody else has done." In this sense, using git pull with the --rebase flag is even more like svn update than a plain git pull.

In fact, pulling with --rebase is such a common workflow that there is a dedicated configuration option for it:

git config --global branch.autosetuprebase always
After running that command, all git pull commands will integrate via git rebase instead of git merge.

eventSec'])
    return df


def parse_contest(contest):
    ignored_columns = []

    df = pd.concat([pd.read_json(data + f"events_{contest}.json")])

    print(f"{contest} data loaded")

    ignored_columns += ['eventId','subEventId']

    names = pd.read_csv(data+"tags2name.csv")
    names.index = names['Tag']
    df['tags'] = df['tags'].apply(lambda ts : [names['Description'][t['id']] for t in ts])

    players = pd.read_json(data+"players.json")
    df = df.merge(players,how='left',left_on='playerId',right_on='wyId').drop('wyId',axis=1)

    df['player'] = df['playerId'].apply(str) + " - " + df['firstName'] + " " + df['middleName'] + " " + df['lastName']
    
    ignored_columns += ['playerId','firstName','middleName','lastName']
    
    matches = pd.concat([pd.read_json(data+f"matches_{competition}.json") for competition in competitions])
    df = df.merge(matches,how='left',left_on='matchId',right_on='wyId').drop('wyId',axis=1)

    df['teamData'] = df.apply(lambda r : r.teamsData[str(r.teamId)],axis=1)
    df['positions'] = df['positions'].apply(lambda t : t[0])
    df['positions'] = df.apply(lambda r : (r.positions['x'],r.positions['y']) if r.teamData['side']=='home' else (100-r.positions['x'],100-r.positions['y']) if r.teamData['side']=='away' else None,axis=1)
    df['positions'] = df['positions'].apply(lambda t : (105*t[0]/100,68*t[1]/100))

    ignored_columns += ['teamData']

    #df['date'] = df['date'].apply(parser.parse)
    #df['birthDate'] = df['birthDate'].apply(lambda x : None if pd.isna(x) else parser.parse(x))
    #df['age'] = df['date']-df['birthDate']

    ignored_columns += ['birthDate']

    df = df.drop(ignored_columns,axis=1)
    df = df.sort_values(['matchId','eventSec'])

    print(f"{contest} data processed")
    return df


def parse_and_save(replace=False):
    for contest in competitions:
        df: pd.DataFrame = parse_contest(contest)
        df.to_csv(f"{data2}/parsed_{contest}.csv")
    

if __name__ == "__main__":
    parse_and_save()
    
