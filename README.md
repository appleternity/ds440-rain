# ds440-rain
This repository contains work adapted from Nikita Saxena in her repository: https://github.com/nikita-0209/downscale-sst
The code was adapted by Ben Elsenheimer and Chieh-Yang Huang for the data science capstone at Pennsylvania State University, under the supervision of Dr. Prasenjit Mitra.

## Data
The raw data for this project can be found at the following OneDrive link: https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/bke5075_psu_edu/EWF6I3wLGdxBg5QN4mYXbfYBA2rNK-GpB5GnNxoynhr9tg?e=f6j8HZ.
However, the intermediate files are included in the Data folder, so raw data may not be necessary to download, especially due to the size (~20GB).

## Create Patches
This file has three main components: extract data from the netCDF files and normalize by the maximum value, upscale the low resolution data so that it can be used with the high resolution, and finally etracting patches of size (33,33) for the actual training. 

Each of these steps is a function, so they can be run sequentially by un-commenting the desired function at the bottom of the file. The build_norm() function uses a range of years provided to extract data from the appropriate files, as well as create an offset for the subset of days contained within the range.

The subsequent functions take the created offset as a parameter, and each step saves its progress as a joblib file to be used by the next step. As stated in the Data section, these intermediate joblib files are included in the repository.

## Train Model
The model can be trained by typing the following into the command line:
#### python3 vdsr_server.py --file_name_low {path to low patches file} --file_name_high {path to high patches file}
The results of the model training are saved as checkpoint files, which are then loaded in the prediction script.
