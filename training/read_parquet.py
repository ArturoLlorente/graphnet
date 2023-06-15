import pandas as pd
import sqlite3


database = '/remote/ceph/user/l/llorente/kaggle/train_meta.parquet'
database_sq = '/remote/ceph/user/l/llorente/kaggle/databases_merged/batch_01.db'

a = pd.read_parquet(database)

print(a)

with sqlite3.connect(database_sq) as con:
    cursor = con.cursor()
    
    # Get the list of tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    # Print the table names
    for table in tables:
        print(table[0])
    query = f'select * from meta_table' # where event_no in {str(tuple(df["event_no"]))}'
    truth = pd.read_sql(query,con)#.sort_values('event_id').reset_index(drop = True)
    
    print(truth)