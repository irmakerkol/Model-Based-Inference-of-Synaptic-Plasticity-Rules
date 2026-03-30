"""
Process raw Rajagopalan et al. (2023) fly behavioral data into the format
expected by Mehta et al. (2024) MetaLearnPlasticity code.

Raw data source: Zenodo DOI 10.5281/zenodo.7449214
Folder: Fig2_Fig4_SuppFig2_SuppFig3_G_SuppFig7/ (18 Gr64f flies, 3-block task)

Input per fly folder (3 blocks):
  - odor_crossing_{1,2,3}.mat: timestamped zone crossings (AtoM, AtoO, etc.)
  - choice_order.mat: (80, 3) which odor accepted per trial per block
  - reward_order.mat: (80, 3) reward code per trial per block

Output per fly (Fly{N}.mat):
  - X: (n_timesteps, 2) one-hot odor identity at EVERY timestep (reject + accept)
  - Y: (n_timesteps,) binary: 0=reject, 1=accept
  - R: (n_trials,) binary reward per completed trial (accept only)

This matches the paper's data_loader.py load_fly_expdata format exactly:
  - Variable-length trials (multiple rejects before each accept)
  - Y contains both 0s and 1s
  - R has one entry per accept (num_trials = sum(Y))

Usage:
  python process_fly_data.py --input_dir /path/to/Fig2_Fig4.../  --output_dir ./data/
"""

import os
import argparse
import numpy as np
import scipy.io as sio


def parse_odor_crossings(filepath):
    """Parse odor crossing events from a .mat file.
    
    Returns list of (time, odor_id) for each entry into an odor zone.
    odor_id: 1 = M (MCH), 2 = O (OCT)
    
    Zone transition types:
      AtoM, OtoM = entered M (odor 1) zone
      AtoO, MtoO = entered O (odor 2) zone
      MtoA, OtoA = returned to air (not an odor entry)
    """
    oc = sio.loadmat(filepath)['odor_crossing']
    entries = []
    
    for i in range(oc.shape[1]):
        entry = oc[0, i]
        time_val = entry['time'].flatten()[0]
        
        # Extract type string (handles nested arrays)
        type_raw = entry['type']
        if hasattr(type_raw, 'flatten'):
            t_arr = type_raw.flatten()
            if len(t_arr) > 0:
                val = t_arr[0]
                if hasattr(val, 'flatten'):
                    val = val.flatten()[0]
                type_str = str(val)
            else:
                type_str = '?'
        else:
            type_str = str(type_raw)
        
        # Only count entries INTO odor zones (not returns to air)
        if type_str in ('AtoM', 'OtoM'):
            entries.append((time_val, 1))  # M = odor 1
        elif type_str in ('AtoO', 'MtoO'):
            entries.append((time_val, 2))  # O = odor 2
    
    return entries


def reconstruct_block(odor_entries, choices_block):
    """Reconstruct full decision sequence for one block.
    
    Each odor zone entry is a timestep. If the odor matches the next
    accepted choice, it's an accept (Y=1). Otherwise it's a reject (Y=0).
    
    Returns:
        full_X: list of odor IDs (1 or 2) per timestep
        full_Y: list of decisions (0=reject, 1=accept) per timestep
        n_consumed: number of accepted trials matched
    """
    trial_idx = 0
    full_X = []
    full_Y = []
    
    for _, odor in odor_entries:
        full_X.append(odor)
        if trial_idx < len(choices_block) and odor == choices_block[trial_idx]:
            full_Y.append(1)  # accept
            trial_idx += 1
        else:
            full_Y.append(0)  # reject
    
    return full_X, full_Y, trial_idx


def process_single_fly(fly_folder, fly_id, output_dir):
    """Process one fly folder into Mehta format with variable-length trials."""
    
    # Check required files exist
    choice_path = os.path.join(fly_folder, 'choice_order.mat')
    reward_path = os.path.join(fly_folder, 'reward_order.mat')
    
    if not os.path.exists(choice_path) or not os.path.exists(reward_path):
        print(f"  WARNING: Missing choice/reward files in {fly_folder}, skipping.")
        return False
    
    # Check odor_crossing files for all 3 blocks
    for b in range(1, 4):
        oc_path = os.path.join(fly_folder, f'odor_crossing_{b}.mat')
        if not os.path.exists(oc_path):
            print(f"  WARNING: Missing odor_crossing_{b}.mat in {fly_folder}, skipping.")
            return False
    
    # Load choice and reward data
    choice_order = sio.loadmat(choice_path)['choice_order']  # (80, 3)
    reward_order = sio.loadmat(reward_path)['reward_order']  # (80, 3)
    
    # Validate shapes
    if choice_order.shape[1] != 3 or reward_order.shape[1] != 3:
        print(f"  WARNING: Unexpected shapes in {fly_folder}, skipping.")
        return False
    
    # Reconstruct full sequence across all 3 blocks
    all_X = []
    all_Y = []
    all_R = []
    total_accepts = 0
    total_rejects = 0
    
    for block in range(3):
        block_num = block + 1
        oc_path = os.path.join(fly_folder, f'odor_crossing_{block_num}.mat')
        
        odor_entries = parse_odor_crossings(oc_path)
        choices = choice_order[:, block]
        rewards = reward_order[:, block]
        
        # Handle choice_order=0 (no choice made)
        valid_choices = choices[choices > 0]
        valid_rewards = rewards[choices > 0]
        
        X_block, Y_block, consumed = reconstruct_block(odor_entries, valid_choices)
        R_block = (valid_rewards > 0).astype(float)
        
        if consumed != len(valid_choices):
            print(f"  WARNING: Block {block_num} mismatch: consumed {consumed}/{len(valid_choices)} trials")
        
        all_X.extend(X_block)
        all_Y.extend(Y_block)
        all_R.extend(R_block.tolist())
        total_accepts += sum(Y_block)
        total_rejects += len(Y_block) - sum(Y_block)
    
    # Convert to arrays
    n_timesteps = len(all_X)
    X_onehot = np.zeros((n_timesteps, 2), dtype=np.float64)
    for i, odor in enumerate(all_X):
        X_onehot[i, int(odor) - 1] = 1.0
    
    Y_arr = np.array(all_Y, dtype=np.float64)
    R_arr = np.array(all_R, dtype=np.float64)
    
    # Verify consistency
    assert int(Y_arr.sum()) == len(R_arr), \
        f"Mismatch: sum(Y)={int(Y_arr.sum())} != len(R)={len(R_arr)}"
    
    # Save
    output_path = os.path.join(output_dir, f'Fly{fly_id}.mat')
    sio.savemat(output_path, {'X': X_onehot, 'Y': Y_arr, 'R': R_arr})
    
    print(f"  Fly{fly_id}: {n_timesteps} timesteps, "
          f"{total_accepts} accepts, {total_rejects} rejects, "
          f"reward rate={R_arr.mean():.3f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Process raw fly data into Mehta format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to Fig2_Fig4_SuppFig2_SuppFig3_G_SuppFig7/ folder')
    parser.add_argument('--output_dir', type=str, default='./data/',
                        help='Output directory for processed Fly{N}.mat files')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    fly_folders = sorted([
        os.path.join(args.input_dir, d)
        for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])
    
    print(f"Found {len(fly_folders)} fly folders in {args.input_dir}")
    print(f"Output directory: {args.output_dir}\n")
    
    fly_id = 1
    processed = 0
    for folder in fly_folders:
        folder_name = os.path.basename(folder)
        print(f"Processing [{fly_id}]: {folder_name}")
        
        if process_single_fly(folder, fly_id, args.output_dir):
            processed += 1
            fly_id += 1
    
    print(f"\nDone! Processed {processed} flies -> {args.output_dir}")
    print(f"Files: Fly1.mat through Fly{processed}.mat")
    print(f"\nFormat per file:")
    print(f"  X: (n_timesteps, 2) one-hot odor at every timestep")
    print(f"  Y: (n_timesteps,) 0=reject, 1=accept")
    print(f"  R: (n_trials,) reward per accepted trial, where n_trials = sum(Y)")


if __name__ == '__main__':
    main()
