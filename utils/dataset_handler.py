# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import os

import nrrd
import scipy.misc
import nibabel as nib

import pandas as pd
from tqdm import tqdm


class DatasetProcessor:
    def __init__(self):
        self.fault_record = []
        self.target_path = 'dataset'
        self.root_data_dir = 'data/annotated_data/slicer'
        self.folder_paths = os.listdir('data/annotated_data/slicer')

    def validate_records(self):
        valid_nii_records_paths, valid_nrrd_records_paths = [], []
        for folder in self.folder_paths:  # 1, 2, 3 ...
            source_folder_path = os.path.join(self.root_data_dir, folder)
            source_files_paths = os.listdir(source_folder_path)

            if len(source_files_paths) != 2:
                self.fault_record.append(source_folder_path)
                continue

            extensions = []
            for file in source_files_paths:
                extensions.append (file.split ('.')[-1])

            if set(extensions) != {'nii', 'nrrd'}:
                self.fault_record.append(source_folder_path)
                continue

            for file in source_files_paths:
                if file.endswith ('nii'):
                    valid_nii_records_paths.append(os.path.join(source_folder_path, file))
                else:
                    valid_nrrd_records_paths.append(os.path.join(source_folder_path, file))

        return sorted(valid_nii_records_paths), sorted(valid_nrrd_records_paths)

    def process_nii(self, original_path, target_folder, index):
        rows = []
        try:
            nii_record = nib.load(original_path)
            record_data = nii_record.get_fdata ()
            image_dim = len(record_data.shape)

            slices_number = record_data.shape[-1]  # e.g.: 54
            height, width = record_data.shape[0], record_data.shape[1]  # e.g.: 128, 128
            slice_resolution = f'{height}*{width}'

            if image_dim == 3:
                for slice_index in range(slices_number):
                    slice_context = record_data[:, :, slice_index]
                    scipy.misc.imsave(f'{target_folder}/{slice_index}_original.jpg', slice_context)

                    buffer_row = {
                        'record_index': index,
                        'record_path': target_folder,
                        'record_class': 'nii',
                        'slice_index': slice_index,
                        'slice_context': slice_context,
                        'slice_resolution': slice_resolution,
                        'slice_size': slice_context.shape,
                    }
                    rows.append(buffer_row)
            return rows

        except Exception as e:
            print(f'[SILENCED] Error: {e} occurred for {original_path}')

    def process_nrrd(self, original_path, target_folder, index):
        rows = []
        try:
            nrrd_data, _ = nrrd.read(original_path)
            image_channels = len(nrrd_data.shape)
            slices_number = nrrd_data.shape[2]  # e.g.: 54
            width, height = nrrd_data.shape[0], nrrd_data.shape[1]  # e.g.: 128, 128
            slice_resolution = f'{width}*{height}'

            if image_channels == 3:
                for slice_index in range(slices_number):
                    slice_context = nrrd_data[:, :, slice_index]
                    scipy.misc.imsave(f'{target_folder}/{slice_index}_segmented.jpg', slice_context)

                    buffer_row = {
                        'record_index': index,
                        'record_path': target_folder,
                        'record_class': 'nrrd',
                        'slice_index': slice_index,
                        'slice_context': slice_context,
                        'slice_resolution': slice_resolution,
                        'slice_size': slice_context.shape,
                    }

                    rows.append(buffer_row)
            return rows

        except Exception as e:
            print(f'[SILENCED] Error: {e} occurred for {original_path}')

    def generate_dataset(self, nii_paths, nrrd_paths):
        for index, (nii, nrrd) in enumerate(tqdm(zip(nii_paths, nrrd_paths))):

            target_folder_path = f'{self.target_path}/{index}'
            target_folder_original = f'{target_folder_path}/original'
            target_folder_segmented = f'{target_folder_path}/segmented'

            os.mkdir(target_folder_path)
            os.mkdir(target_folder_original)
            os.mkdir(target_folder_segmented)

            nii_rows = self.process_nii(original_path=nii, target_folder=target_folder_original, index=index)
            nrrd_rows = self.process_nrrd(original_path=nrrd, target_folder=target_folder_segmented, index=index)

            result_df = pd.DataFrame

    def run(self):
        nii_records, nrrd_records = self.validate_records()
        self.generate_dataset(nii_paths=nii_records, nrrd_paths=nrrd_records)



