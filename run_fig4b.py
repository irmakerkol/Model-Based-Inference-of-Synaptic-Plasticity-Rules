"""Figure 4B: Example fly behavior raster plot."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.io as sio
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_fly_raster(fly_id=1, data_dir='./data/', save_path='figures/figure4b.png'):
    d = sio.loadmat(os.path.join(data_dir, f'Fly{fly_id}.mat'))
    X = d['X']; Y = d['Y'].flatten(); R = d['R'].flatten()
    trial_odors, trial_rewards = [], []; tidx = 0
    for t in range(len(Y)):
        if Y[t] > 0.5:
            odor = 0 if X[t, 0] > X[t, 1] else 1
            trial_odors.append(odor)
            trial_rewards.append(R[tidx] if tidx < len(R) else 0)
            tidx += 1
    n = len(trial_odors); w = 10
    cr = [sum(1 for o in trial_odors[max(0,i-w+1):i+1] if o==0)/(i-max(0,i-w+1)+1) for i in range(n)]
    rr = [np.mean(trial_rewards[max(0,i-w+1):i+1]) for i in range(n)]
    
    fig, ax = plt.subplots(figsize=(14, 4))
    for i in range(n):
        y_pos = 1.0 if trial_odors[i]==0 else 0.0
        h = 0.3 if trial_rewards[i]>0.5 else 0.15
        c = 'tab:blue' if trial_odors[i]==0 else 'tab:orange'
        ax.plot([i,i],[y_pos-h,y_pos+h], color=c, alpha=0.9 if trial_rewards[i]>0.5 else 0.4, lw=1)
    ax.plot(range(n), cr, 'r-', lw=2, label='Choice ratio', alpha=0.8)
    ax.plot(range(n), rr, 'k-', lw=1.5, label='Reward ratio', alpha=0.6)
    for b in [80,160]:
        if b<n: ax.axvline(b, color='gray', ls='--', lw=1)
    for b,lb in enumerate(['80:20','11:89','89:11']):
        mid = b*80+40
        if mid<n: ax.text(mid, 1.45, lb, ha='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Trial Number'); ax.set_ylabel('Odor')
    ax.set_yticks([0,1]); ax.set_yticklabels(['Odor 2','Odor 1'])
    ax.set_xlim(-2, min(n,245)); ax.set_ylim(-0.5,1.6)
    ax.set_title(f'Figure 4B: Behavior of Fly {fly_id}', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.text(0.01,0.95,'Tall=rewarded, Short=unrewarded',transform=ax.transAxes,fontsize=8,va='top',
            bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.5))
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved {save_path}")

if __name__ == '__main__':
    for fid in [1,2,5]:
        if os.path.exists(f'./data/Fly{fid}.mat'):
            plot_fly_raster(fid, save_path=f'figures/figure4b_fly{fid}.png')
