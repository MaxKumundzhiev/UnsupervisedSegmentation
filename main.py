# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import os

import pandas as pd
from tqdm import tqdm

from utils.dataset_handler import DatasetProcessor


if __name__ == '__main__':
    # df = DatasetProcessor().run()

    # @TODO resolve logic issuing for model call
    df = pd.read_csv('patients.csv')
    original_paths = list(set(df['original_record_path'].values))

    for patient_record_path in original_paths:
        print(f'[PROCESSING]: {patient_record_path}')

        patients_slices_paths = os.listdir(patient_record_path)
        patients_slices_paths = [os.path.join(patient_record_path, path) for path in patients_slices_paths]

        # creating of predicted folder
        target_path = patient_record_path.replace('original', 'predicted')
        if not os.path.exists(target_path):
            os.mkdir(target_path)

        for _, slice_path in enumerate(tqdm(patients_slices_paths)):
            cmd = f'python predict.py --input {slice_path}'
            os.system(cmd)
