# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import os
from tqdm import tqdm

import nibabel as nib


class DatasetProcessor:
    def __init__(self):
        self.root_data_dir = 'data/annotated_data/slicer'
        self.folder_paths = os.listdir('data/annotated_data/slicer')
        self.fault_record = []

    def validate_records(self):
        valid_nii_records_paths, valid_nrdd_records_paths = [], []
        for folder in self.folder_paths:  # 1, 2, 3 ...
            source_folder_path = os.path.join(self.root_data_dir, folder)
            source_files_paths = os.listdir(source_folder_path)

            if len (source_files_paths) != 2:
                self.fault_record.append(source_folder_path)
                continue

            extensions = []
            for file in source_files_paths:
                extensions.append (file.split ('.')[-1])

            if set (extensions) != {'nii', 'nrrd'}:
                self.fault_record.append(source_folder_path)
                continue

            for file in source_files_paths:
                if file.endswith ('nii'):
                    valid_nii_records_paths.append(os.path.join(source_folder_path, file))
                else:
                    valid_nrdd_records_paths.append(os.path.join(source_folder_path, file))

        return valid_nii_records_paths, valid_nrdd_records_paths


    def preprocess_dataset(self):
        nii_records, nrdd_records = validate_records()


    #
    # @staticmethod
    # def get_files(full_folder_path):
    #     files = os.listdir(full_folder_path)
    #     return f'{full_folder_path}/{files[0]}', f'{full_folder_path}/{files[1]}'
    #
    # @staticmethod
    # def capture_nii_image(nii_record_path):
    #     rows = []
    #
    #     try:
    #         record = nib.load(nii_record_path)
    #         record_data = record.get_fdata()  # (128, 128, 54), where 54 denotes number of slices relied to the image
    #         image_dim = len(record_data.shape)
    #
    #         slices_number = record_data.shape[-1]  # e.g.: 54
    #         height, width = record_data.shape[0], record_data.shape[1]  # e.g.: 128, 128
    #         slice_resolution = f'{height}*{width}'
    #
    #         if image_dim == 3:
    #             for slice_index in range(slices_number):
    #                 slice_context = record_data[:, :, slice_index]
    #                 buffer_row = {
    #                     'record_path': nii_record_path,
    #                     'record_class': 'nii',
    #                     'slices_index': slice_index,
    #                     'slice_context': slice_context,
    #                     'slice_resolution': slice_resolution,
    #                     'slice_size': slice_context.shape,
    #                 }
    #                 rows.append(buffer_row)
    #             return rows
    #         else:
    #             raise
    #
    #     except Exception as e:
    #         print(f'[SILENCED] Error: {e} occurred for {nii_record_path}')
    #
    # def run(self):
    #     failed_folders = []
    #     rows = []
    #
    #     for _, folder in enumerate(tqdm(self.folder_paths)):
    #         full_folder_path = os.path.join(self.root_data_dir, folder)
    #         files = os.listdir(full_folder_path)
    #
    #         # check whether folder consist of expected .nii and .nrdd files
    #         if len(files) < 2:
    #             failed_folders.append(full_folder_path)
    #             continue
    #
    #         # elif not [file.endswith('.nii') for file in files] and not [file.endswith('.nrdd') for file in files]:
    #         #     failed_folders.append(full_folder_path)
    #         #     continue
    #
    #         else:
    #             if self.rename_flag:
    #                 original_image_path, segmented_image_path = self.rename_files(full_folder_path, files)
    #             else:
    #                 original_image_path, segmented_image_path = self.get_files(full_folder_path)
    #
    #             buff_nii_rows = self.capture_nii_image(nii_record_path=original_image_path)
    #             rows.append(buff_nii_rows)
    #
    #     return rows, failed_folders
    #
