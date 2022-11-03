#%%
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

#%%
# get all the image folder paths
all_paths = os.listdir('C:/CCBDA/HW2/test')
folder_paths = [x for x in all_paths if os.path.isdir('C:/CCBDA/HW2/test/' + x)]
folder_paths = [str(i) for i in folder_paths]
print(f"Folder paths: {folder_paths}")
print(f"Number of folders: {len(folder_paths)}")

#%%
# create a dataframe to store image path
data = pd.DataFrame()

#%%

image_formats = ['jpg'] # we only want images that are in this format
labels = []
counter = 0
for i, folder_path in tqdm(enumerate(folder_paths), total=len(folder_paths)):
    image_paths = os.listdir('C:/CCBDA/HW2/test/'+folder_path)
    label = folder_path
    # save image paths in the DataFrame
    for image_path in image_paths:
        if image_path.split('.')[-1] in image_formats:
            data.loc[counter, 'image_path'] = f"C:/CCBDA/HW2/test/{folder_path}/{image_path}"
            labels.append(label)
            counter += 1

#%%
if len(labels[0]) == 1:
    for i in range(len(labels)):
        index = labels[i]
        data.loc[i, 'target'] = int(index)
elif len(labels[0]) > 1:
    for i in range(len(labels)):
        index = np.argmax(labels[i])
        data.loc[i, 'target'] = int(index)
#%%
# shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)
print(f"Total instances: {len(data)}")

# save as CSV file
data.to_csv('C:/CCBDA/HW2/data_test.csv', index=False)

print(data.head(5))
# %%
