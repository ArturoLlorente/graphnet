import pandas as pd
import sqlite3
import os
import numpy as np
from tqdm import tqdm


with sqlite3.connect('/scratch/users/allorana/cascades_21537.db') as conn:
    df = pd.read_sql('SELECT event_no,energy FROM truth', conn)
    print('truth table read')
    df_pulses = pd.read_sql('SELECT event_no FROM InIceDSTPulses', conn)
    print('pulses table read')

unique, pulses_count = np.unique(np.array(df_pulses["event_no"]), return_counts=True)


energies = np.array([df.loc[df['event_no']==ev]['energy'] for ev in tqdm(unique)]).squeeze()
#print(energies)
#print(energies.shape())

df_all = pd.DataFrame({'event_no': unique, 'n_pulses': pulses_count, 'energy': energies})
df_all.to_csv('/scratch/users/allorana/cascades_21537_selection_plus_energy.csv', index=False)
