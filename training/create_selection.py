import pandas as pd
import sqlite3
import os
import numpy as np
from tqdm import tqdm


db_path = '/scratch/users/allorana/merged_sqlite_1505/'
for i in tqdm(range(34)):
    filepath = os.path.join(db_path, f'part{i}/merged/merged.db')
    print(filepath)
    with sqlite3.connect(filepath) as conn:
        df = pd.read_sql('SELECT event_no,energy FROM truth', conn)
        df_pulses = pd.read_sql('SELECT event_no FROM InIceDSTPulses', conn)
        #print('pulses table read')

    unique, pulses_count = np.unique(np.array(df_pulses["event_no"]), return_counts=True)

    energies = np.array([df.loc[df['event_no']==ev]['energy'] for ev in unique]).squeeze()

    df_all = pd.DataFrame({'event_no': unique, 'n_pulses': pulses_count, 'energy': energies})
    df_all.to_csv(f'/scratch/users/allorana/merged_sqlite_1505/selection_files/part{i}_n_pulses.csv', index=False)
    
    del df, df_pulses, df_all, unique, pulses_count, energies
