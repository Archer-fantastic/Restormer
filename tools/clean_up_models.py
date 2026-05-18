import os
import glob
import re

def clean_up(experiment_dir):
    models_dir = os.path.join(experiment_dir, 'models')
    states_dir = os.path.join(experiment_dir, 'training_states')
    
    # 1. Clean models
    pth_files = glob.glob(os.path.join(models_dir, 'net_g_*.pth'))
    # Extract iteration numbers
    iter_files = []
    for f in pth_files:
        basename = os.path.basename(f)
        match = re.search(r'net_g_(\d+)\.pth', basename)
        if match:
            iter_files.append((int(match.group(1)), f))
            
    if iter_files:
        # Sort by iteration number
        iter_files.sort(key=lambda x: x[0])
        # Keep the latest one
        latest_iter, latest_file = iter_files[-1]
        
        # Delete older ones
        for it, f in iter_files[:-1]:
            print(f"Deleting redundant model: {f}")
            os.remove(f)
            
    # 2. Clean training states
    state_files = glob.glob(os.path.join(states_dir, '*.state'))
    # Extract iteration numbers
    iter_states = []
    for f in state_files:
        basename = os.path.basename(f)
        match = re.search(r'^(\d+)\.state$', basename)
        if match:
            iter_states.append((int(match.group(1)), f))
            
    if iter_states:
        # Sort by iteration number
        iter_states.sort(key=lambda x: x[0])
        # Keep the latest one
        latest_iter, latest_file = iter_states[-1]
        
        # Delete older ones
        for it, f in iter_states[:-1]:
            print(f"Deleting redundant state: {f}")
            os.remove(f)

if __name__ == '__main__':
    clean_up(r'experiments\DCK_384_25')
