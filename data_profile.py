#%%

# import os

# def count_dicoms(directory):
#     count = 0
#     # Walk through all subdirectories
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.lower().endswith('.dcm'):
#                 count += 1
#     return count

# root_dir = "rsna_intracranial/rsna-intracranial-hemorrhage-detection"
# train_folder = "stage_2_train"
# test_folder = "stage_2_test"
# train_dir = os.path.join(root_dir, train_folder)
# test_dir = os.path.join(root_dir, test_folder)

# print("Number of DICOM files in train:", count_dicoms(train_dir))
# print("Number of DICOM files in test:", count_dicoms(test_dir))

# # Number of DICOM files in train: 752803
# # Number of DICOM files in test: 121232



# #%%

# import pydicom
# import matplotlib.pyplot as plt

# root_dir = "rsna_intracranial/rsna-intracranial-hemorrhage-detection"
# train_folder = "stage_2_train"
# test_folder = "stage_2_test"
# train_dir = os.path.join(root_dir, train_folder)
# test_dir = os.path.join(root_dir, test_folder)

# #dicom_file = "ID_4cba40133.dcm"
# dicom_file = "ID_4cba2f29c.dcm"
# dicom_path = os.path.join(train_dir, dicom_file)
# dicom = pydicom.dcmread(dicom_path)



# print("Series ID:",dicom[0x0020, 0x000E].value)
# print("Study ID:",dicom[0x0020, 0x000D].value)
# print("Patient ID:",dicom[0x0010, 0x0020].value)



#%%

import os
import pydicom
from tqdm import tqdm
import json

def collect_dicom_metadata(directory):
    metadata = {}
    for filename in tqdm(os.listdir(directory)):
        if filename.lower().endswith('.dcm'):
            filepath = os.path.join(directory, filename)
            ds = pydicom.dcmread(filepath)
            # Use attribute names with a fallback if not present
            series_id = ds[0x0020, 0x000E].value if (0x0020, 0x000E) in ds else "N/A"
            study_id = ds[0x0020, 0x000D].value if (0x0020, 0x000D) in ds else "N/A"
            patient_id = ds[0x0010, 0x0020].value if (0x0010, 0x0020) in ds else "N/A"
            metadata[filename] = {
                "SeriesID": series_id,
                "StudyID": study_id,
                "PatientID": patient_id
            }
    return metadata

# Example usage:
train_dir = "rsna_intracranial/rsna-intracranial-hemorrhage-detection/stage_2_test"
dicom_metadata = collect_dicom_metadata(train_dir)

# Now, dicom_metadata is a dictionary where each key is a filename, and the value is a dict containing metadata.

# Save the metadata to a JSON file
save_path = "dicom_metadata_test.json"
with (open(save_path, "w")) as f:
    json.dump(dicom_metadata, f)


#%%

# import json
# import pandas as pd
# from tqdm import tqdm

# # Load the metadata from the JSON file

# load_path = "dicom_metadata.json"
# with open(load_path, "r") as f:
#     dicom_metadata = json.load(f)

# records = []
# for filename, meta in tqdm(dicom_metadata.items()):
#     record = {
#         "filename": filename,
#         "SeriesID": meta["SeriesID"],
#         "StudyID": meta["StudyID"],
#         "PatientID": meta["PatientID"]
#     }
#     records.append(record)

# df = pd.DataFrame(records)
# df.head()

# #%%

# df_grouped = df.groupby("PatientID").agg(
#     SeriesID_CNT=("SeriesID", "nunique"),
#     StudyID_CNT=("StudyID", "nunique"),
#     filename_CNT=("filename", "nunique"),
#     CNT=("filename", "count")
# ).reset_index()

# print(df_grouped)

# #%%

# df_grouped.head(15)



# #%%

# filtered_df = df_grouped[df_grouped['SeriesID_CNT'] > 1]
# print(filtered_df)

# patientid_cnt = df_grouped[df_grouped['SeriesID_CNT'] > 1].shape[0]
# print("PatientID_CNT:", patientid_cnt)

# #%%

# filtered_df.describe()



#%%