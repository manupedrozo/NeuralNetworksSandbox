import os
from datetime import datetime

def create_csv(results, model_name, results_dir='./results'):
    """Generate csv for submission"""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    csv_fname = f'results_{model_name}_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:

        f.write('Id,Category\n')

        for key, value in results.items():
            f.write(key + ',' + str(value) + '\n')

def get_files_in_directory(path, include_folders=False):
    """Get all filenames in a given directory, optionally include folders as well"""
    return [f for f in os.listdir(path) if include_folders or os.path.isfile(os.path.join(path, f))]