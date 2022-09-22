import matplotlib.pyplot as plt
import numpy as np
import csv
returns=[]
episode_len=[]


with open('returns.txt', 'r') as datafile:
    plotting = csv.reader(datafile)
    
    for ROW in plotting:
        returns.append(float(ROW[0]))
        episode_len.append(float(ROW[1]))


n_eps = len(returns)
eps = range(1,n_eps+1)
 
plt.figure()
plt.plot(eps,returns, color="orange",lw=2, alpha=0.1, label="Lighten")
avg_returns_cal=[]
for i in range(1,n_eps+1):
    avg_returns_cal.append(np.mean(returns[i-min(i,50):i+1]))
plt.plot(eps,avg_returns_cal, color="orange")
plt.xlabel('episodes')
plt.ylabel('game_rewards')
plt.savefig('game_rewards.png')


plt.figure()
plt.plot(eps, episode_len, color="orange",lw=2, alpha=0.1, label="Lighten")
avg_episode_len=[]
for i in range(1,n_eps+1):
    avg_episode_len.append(np.mean(episode_len[i-min(i,50):i+1]))
plt.plot(eps,avg_episode_len, color="orange")
plt.xlabel('episodes')
plt.ylabel('episode_len')
plt.savefig('episode_len.png')
