import numpy as np
import glob
import heapq
import re

OUTPUT_DIR="runs/detect/"
videos = glob.glob(OUTPUT_DIR+"/*")

get_index = lambda filename: re.search(r"\d_(\d+).txt", filename).groups()[0]
get_filename = lambda filename: re.search(
    r"(\w+_\w+-\w+-\w+)_\w+.txt", filename
).groups()[0]

for video in videos:
    data = []
    files = glob.glob(video+"/labels/*")
    print(f"Processing video: {video}")
    if len(files) == 0:
        print(f"No detection for {video} skipping")
        continue
    print(f"Detected {len(files)} files")
    for file in files:
        index = int(get_index(file))

        with open(file, 'r') as f:
            for line in f:
                line = line.replace("\n", "")
                elements = line.split(" ")
                new_entry = np.full((1, 6), np.nan)
                new_entry[0,0] = index
                for i in range(1,6):
                    new_entry[0, i] = float(elements[i])
                data.append(new_entry)
    filename = get_filename(files[0])
    data = np.concatenate(data)
    np.save(f"{filename}_roach.npy", data)
