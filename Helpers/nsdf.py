import os
import subprocess
from tqdm import tqdm
from pathlib import Path

def get_file_paths(base_folder):
    base_folder = base_folder
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file[-4:] == ".vdb":
                file_path = os.path.join(root, file)
                yield file_path

def sample_sdf(file, num_points=10000):
    command = ["./nsdf", file, str(num_points)]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error Processing {file}: {e}")

file_path_gen = get_file_paths("./Data/vdbs/")
count = 0
for root, dirs, files in os.walk('./Data/vdbs/'):
    for file in files:
        if file[-4:] == ".vdb":
            count += 1
    

#total_files = sum([len(files) for _, _, files in os.walk(str(Path.home()) + '/dev/data/vdb/vdbs/')])
print("Total Files: ", count)

for file in tqdm(file_path_gen, total=count):
    sample_sdf(file)
    

