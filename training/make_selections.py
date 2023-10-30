import pandas as pd
import sqlite3
from tqdm import tqdm

path = '/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/'
databases = ['dev_northern_tracks_muon_labels_v3_part_1.db',  
             'dev_northern_tracks_muon_labels_v3_part_2.db',
             'dev_northern_tracks_muon_labels_v3_part_3.db', 
             'dev_northern_tracks_muon_labels_v3_part_4.db', 
             'dev_northern_tracks_muon_labels_v3_part_5.db',
             'dev_northern_tracks_muon_labels_v3_part_6.db', 
             'dev_northern_tracks_muon_labels_v3_part_7.db', 
             'dev_northern_tracks_muon_labels_v3_part_8.db',]
             
pulsemap = 'InIceDSTPulses'

for database in databases:
    with sqlite3.connect(path + database) as con:
        query = 'select event_no from northeren_tracks_muon_labels where '
        df = pd.read_sql(query,con) #.to_csv(database.split('.db')[0] + '_regression_selection.csv')
    
    lengths = []
    for event_no in tqdm(df['event_no']):
        with sqlite3.connect(path + '/' + database) as con:
            query = f'select event_no from {pulsemap} where event_no = {event_no}'
            lengths.append(len(con.execute(query).fetchall()))
    df['n_pulses'] = lengths
    df.to_csv(f'/remote/ceph/user/l/llorente/northern_track_selection/{database[35:-3]}.csv')
    
    print(f'Finished {database}')