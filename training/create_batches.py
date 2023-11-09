import pandas as pd
from tqdm.auto import tqdm

limits = [4000, 200000]
test_selection = pd.read_csv(f'/remote/ceph2/user/l/llorente/northern_track_selection/part_1.csv')
a = test_selection.loc[(test_selection['n_pulses']>=limits[0]) & (test_selection['n_pulses']<limits[1]),:]
f = a['n_pulses'].to_numpy()

def DYNAMICBUCKETING(Q, f):
    n = len(f)
    
    dp = [[float('inf')] * (n + 1) for _ in range(Q + 1)]
    prevDp = [[-1] * (n + 1) for _ in range(Q + 1)]
    
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[0][i] = float('inf')
    
    pbar = tqdm(total=Q*n)
    for q in range(1, Q + 1):
        for i in range(1, n + 1):
            curSum = f[i - 1]
            for j in range(i - 2, -1, -1):
                curSum += f[j]
                val = curSum * i + dp[q - 1][j]
                if val < dp[q][i]:
                    dp[q][i] = val
                    prevDp[q][i] = j
            pbar.update(1)
    
    curId = n - 1
    bests = []
    for i in range(Q, 0, -1):
        bests.insert(0, curId)
        curId = prevDp[i][curId]
    
    return bests

Q = 3
result = DYNAMICBUCKETING(Q, f)
print("Best bucketing indices:", result)