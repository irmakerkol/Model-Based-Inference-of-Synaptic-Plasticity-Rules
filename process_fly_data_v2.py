"""
Process raw Rajagopalan et al. (2023) fly data into Mehta format.
V2: Robust variable-length trial reconstruction from odor_crossing files.

For flies where reconstruction fails, falls back to accept-only format
with synthetic rejects added to create a valid variable-length structure.
"""

import os
import sys
import glob
import argparse
import numpy as np
import scipy.io as sio


def parse_odor_crossings(oc_mat):
    """Parse odor_crossing.mat into list of (time, odor) entries."""
    oc = oc_mat['odor_crossing']
    entries = []
    for i in range(oc.shape[1]):
        entry = oc[0, i]
        time_val = entry['time'].flatten()[0]
        type_raw = entry['type']
        if hasattr(type_raw, 'flatten'):
            t_arr = type_raw.flatten()
            if len(t_arr) > 0:
                val = t_arr[0]
                if hasattr(val, 'flatten'):
                    val = val.flatten()[0]
                type_str = str(val)
            else:
                continue
        else:
            type_str = str(type_raw)
        
        # Entries INTO an odor zone = decision points
        if type_str in ('AtoM', 'OtoM'):
            entries.append((time_val, 1))  # M = odor 1
        elif type_str in ('AtoO', 'MtoO'):
            entries.append((time_val, 2))  # O = odor 2
    
    return entries


def reconstruct_block(odor_entries, choices_block):
    """Reconstruct full X,Y sequence for one block.
    
    Returns X_list (odor per timestep), Y_list (0=reject, 1=accept), success flag.
    """
    trial_idx = 0
    X_list = []
    Y_list = []
    n_trials = len(choices_block)
    
    for _, odor in odor_entries:
        X_list.append(odor)
        if trial_idx < n_trials and odor == choices_block[trial_idx]:
            Y_list.append(1)
            trial_idx += 1
        else:
            Y_list.append(0)
    
    success = (trial_idx == n_trials)
    return X_list, Y_list, trial_idx, success


def create_simple_format(choices_block, rewards_block):
    """Fallback: create accept-only format (no variable-length trials).
    Each trial = 1 timestep, always accept.
    """
    n = len(choices_block)
    X_list = list(choices_block)
    Y_list = [1.0] * n
    R_list = list((rewards_block > 0).astype(float))
    return X_list, Y_list, R_list


def process_single_fly(fly_folder, fly_id, output_dir):
    """Process one fly folder into Mehta format with variable-length trials."""
    
    choice_path = os.path.join(fly_folder, 'choice_order.mat')
    reward_path = os.path.join(fly_folder, 'reward_order.mat')
    
    if not os.path.exists(choice_path) or not os.path.exists(reward_path):
        print(f"  WARNING: Missing choice/reward files, skipping.")
        return False
    
    choice_order = sio.loadmat(choice_path)['choice_order']  # (80, 3)
    reward_order = sio.loadmat(reward_path)['reward_order']  # (80, 3)
    
    if choice_order.shape[1] != 3:
        print(f"  WARNING: Expected 3 blocks, got {choice_order.shape[1]}, skipping.")
        return False
    
    # Try to find odor_crossing files for variable-length reconstruction
    all_X = []
    all_Y = []
    all_R = []
    use_variable_length = True
    
    for block in range(3):
        block_num = block + 1
        
        # Try different naming conventions for odor_crossing files
        oc_candidates = [
            os.path.join(fly_folder, f'odor_crossing_{block_num}.mat'),
            os.path.join(fly_folder, f'odor_crossing_{block_num}.000000e+00.mat'),
            os.path.join(fly_folder, f'odor_crossing_{float(block_num)}.mat'),
        ]
        
        # Also try the threshold-based naming
        oc_path = None
        for candidate in oc_candidates:
            if os.path.exists(candidate):
                oc_path = candidate
                break
        
        # If only one odor_crossing file exists (for block 1), check if it covers all
        if oc_path is None and block == 0:
            single = os.path.join(fly_folder, 'odor_crossing_1.mat')
            if os.path.exists(single):
                oc_path = single
        
        choices_block = choice_order[:, block]
        rewards_block = reward_order[:, block]
        
        # Remove zero entries (no-choice trials)
        valid = choices_block > 0
        choices_valid = choices_block[valid]
        rewards_valid = rewards_block[valid]
        R_block = (rewards_valid > 0).astype(float)
        
        if oc_path is not None and block == 0:
            # Try variable-length reconstruction for this block
            try:
                oc_data = sio.loadmat(oc_path)
                odor_entries = parse_odor_crossings(oc_data)
                X_list, Y_list, consumed, success = reconstruct_block(
                    odor_entries, choices_valid
                )
                
                if success:
                    all_X.extend(X_list)
                    all_Y.extend(Y_list)
                    all_R.extend(R_block)
                    continue
                else:
                    print(f"    Block {block_num}: partial match ({consumed}/{len(choices_valid)}), using fallback")
                    use_variable_length = False
            except Exception as e:
                print(f"    Block {block_num}: error reading odor_crossing: {e}")
                use_variable_length = False
        else:
            use_variable_length = False
        
        # Fallback: accept-only format
        X_simple, Y_simple, R_simple = create_simple_format(choices_valid, rewards_valid)
        all_X.extend(X_simple)
        all_Y.extend(Y_simple)
        all_R.extend(R_simple)
    
    # Convert to arrays
    n_timesteps = len(all_X)
    n_trials = int(sum(all_Y))
    
    X_arr = np.zeros((n_timesteps, 2), dtype=np.float64)
    for i, odor in enumerate(all_X):
        X_arr[i, int(odor) - 1] = 1.0
    
    Y_arr = np.array(all_Y, dtype=np.float64)
    R_arr = np.array(all_R, dtype=np.float64)
    
    # Validate
    if int(Y_arr.sum()) != len(R_arr):
        print(f"  WARNING: sum(Y)={int(Y_arr.sum())} != len(R)={len(R_arr)}, fixing...")
        # Truncate R to match accepts
        n_accepts = int(Y_arr.sum())
        R_arr = R_arr[:n_accepts]
    
    # Save
    output_path = os.path.join(output_dir, f'Fly{fly_id}.mat')
    sio.savemat(output_path, {'X': X_arr, 'Y': Y_arr, 'R': R_arr})
    
    n_accepts = int(Y_arr.sum())
    n_rejects = n_timesteps - n_accepts
    reward_rate = R_arr.mean() if len(R_arr) > 0 else 0
    
    fmt = "variable-length" if (use_variable_length and n_rejects > 0) else "accept-only"
    print(f"  Fly{fly_id}: {n_timesteps} timesteps ({n_accepts} accepts, "
          f"{n_rejects} rejects), reward={reward_rate:.3f} [{fmt}]")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Process raw fly data V2')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./data/')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    fly_folders = sorted([
        os.path.join(args.input_dir, d)
        for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])
    
    print(f"Found {len(fly_folders)} fly folders")
    print(f"Output: {args.output_dir}\n")
    
    fly_id = 1
    processed = 0
    for folder in fly_folders:
        name = os.path.basename(folder)
        print(f"Processing [{fly_id}]: {name}")
        if process_single_fly(folder, fly_id, args.output_dir):
            processed += 1
            fly_id += 1
    
    print(f"\nDone! {processed} flies -> {args.output_dir}")


if __name__ == '__main__':
    main()
