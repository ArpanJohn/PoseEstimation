import os
import json
import nbconvert
import nbformat
import subprocess
from support.funcs import *
import time

# Measure the execution time
start_time = time.time()

current_directory = os.getcwd()
with open("C:\\Users\\CMC\\Downloads\\PoseEstimation\\rec_program\\savdir_path.json") as json_file:
    data = json.load(json_file)
    # Access the directory path
    pth = data['directory_path']

def get_folders_in_directory(directory_path):
    folders = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]
    return folders

# Replace "path/to/your/directory" with the actual path of the directory you want to scan.
directory_path = pth

folders_list = get_folders_in_directory(directory_path)
folders_list = [pth+"\\"+string for string in folders_list if string.startswith('21-07')]
print(folders_list)


# write_to_json.py

for SessDir in folders_list:
    data = {
            "directory": SessDir
            }
    # Read the JSON file containing the Session Directory
    with open(current_directory+'\\upperbody\\SessionDirectory.json', 'w') as file:
        json.dump(data, file)

    # run_python_script.py
    script_filename = "C:\\Users\\CMC\\Downloads\\PoseEstimation\\upperbody\\msg_to_mpipe_from_point.py"
    print('running '+ script_filename.split("\\")[-1] + " on " + data["directory"].split("\\")[-1])
    subprocess.run(["python", script_filename])

    # run_notebook.py

    notebook_filename = "C:\\Users\\CMC\\Downloads\\PoseEstimation\\upperbody\\5.Euler angles interpolated.ipynb"
    print('running '+ notebook_filename.split("\\")[-1] + " on " + data["directory"].split("\\")[-1])
    # Execute the notebook and save the output
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
        nbconvert.preprocessors.execute.ExecutePreprocessor(timeout=600).preprocess(nb)

    notebook_filename = "C:\\Users\\CMC\\Downloads\\PoseEstimation\\upperbody\\more_graphs.ipynb"
    print('running '+ notebook_filename.split("\\")[-1] + " on " + data["directory"].split("\\")[-1])
    # Execute the notebook and save the output
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
        nbconvert.preprocessors.execute.ExecutePreprocessor(timeout=600).preprocess(nb)

# Calculate the elapsed time
elapsed_time = time.time() - start_time

# Print the elapsed time
print(f"Program executed in {elapsed_time:.2f} seconds")