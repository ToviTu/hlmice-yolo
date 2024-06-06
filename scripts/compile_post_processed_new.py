import numpy as np
import glob
import heapq
import re

output_dir = "./yolov5/runs/detect/out/labels/"

row_structure = [0, 1, 2, 3, 3, 4, 5, 6, 7]
available_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]


files = glob.glob(output_dir + "*")
get_index = lambda filename: re.search(r"\d_(\d+).txt", filename).groups()[0]
get_filename = lambda filename: re.search(
    r"(\w+_\w+-\w+-\w+)_\w+.txt", filename
).groups()[0]

videos = set()
for file in files:
    videos.add(get_filename(file))

for video in videos:
    files_video = [file for file in files if video in file]
    sorted_files = []
    for file in files_video:
        score = int(get_index(file))
        heapq.heappush(sorted_files, (score, file))

    data = []
    counter = 0
    number_missing = 0
    while sorted_files != []:
        idx, file = sorted_files.pop(0)
        with open(file, "r") as f:
            line = f.read()
        entries = line.split("\n")[:-1]

        placeholder = np.full(
            ((np.max(available_indices) + 1) * 5), -np.inf
        )  # 4 coordinated for bounding boxes and 1 for confidence
        counter += 1

        # Missing file detection
        if int(idx) != counter:
            data.append(placeholder)
            number_missing += 1
            counter += 1
            print(f"missing frame detected at {counter}")

        # iterate through entries
        detected = []
        for object_data in entries:
            points = object_data.split(" ")
            id = int(points[0])
            start_idx = np.where(np.array(row_structure) == id)[0]

            if id not in available_indices:
                print(f"Found {id} but not in available indices")
                continue

            conf_new = float(points[-1])
            for idx in start_idx:
                conf = placeholder[idx * 5 + 4]
                if len(start_idx) > 1 and conf > -np.inf:
                    #print(f"{conf_new} {conf}")
                    continue
                if conf_new > conf:
                    for i in range(len(points) - 1):
                        placeholder[idx * 5 + i] = float(points[i + 1])
                    detected.append(id)
                    break

        # Special Handeling for the balls
        ball_indices = np.where(np.array(row_structure) == 3)[0]
        if placeholder[ball_indices[0] * 5] != -np.inf and placeholder[ball_indices[1] * 5] != -np.inf:
            if placeholder[ball_indices[0] * 5 + 1] >= placeholder[ball_indices[1] * 5 + 1]:
                temp = placeholder[ball_indices[0] * 5 : ball_indices[0] * 5 + 5].copy()
                placeholder[ball_indices[0] * 5 : ball_indices[0] * 5 + 5] = placeholder[ball_indices[1] * 5 : ball_indices[1] * 5 + 5]
                placeholder[ball_indices[1] * 5 : ball_indices[1] * 5 + 5] = temp

        if set(detected) != set(row_structure):
            set_not_found = set(row_structure) - set(detected)
            print(f"The follow objects are not detected: {set_not_found}")

        placeholder[placeholder == -1] = np.nan
        data.append(placeholder)

    data = np.stack(data)
    assert data.shape[0] > 53900 and data.shape[0] < 54100
    print(f"There are {number_missing} missing frames")
    np.save(f"./{video}_processed_array.npy", data)
