import sys
import warnings
import time as t
import os
from os.path import isfile, join
from itertools import groupby
from operator import itemgetter
import shutil
import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
# plt.switch_backend('Agg')

warnings.filterwarnings("ignore")


def load_preds_file(preds_filename, image_width=640, image_height=480):
    """
    Load the prediction files and un-normalize values
    :param preds_filename: .txt file with object centroid coordinates
    :param image_width: width of image
    :param image_height: height of image
    """
    if preds_filename == ".DS_Store":
        return
    df = pd.read_csv(preds_filename, sep=" ", header=None)
    if len(df.columns) == 6:
        df.columns = [
            "label",
            "x_center",
            "y_center",
            "x_width",
            "y_height",
            "confidence",
        ]
    elif len(df.columns) == 5:
        df.columns = ["label", "x_center", "y_center", "x_width", "y_height"]
        df["confidence"] = [np.nan] * len(df)
    df["x_center"] = df["x_center"] * image_width
    df["y_center"] = df["y_center"] * image_height
    df["x_width"] = df["x_width"] * (image_width / 2)  # used to be just img_width
    df["y_height"] = df["y_height"] * (image_height / 2)
    df["label"] = df["label"].map(
        {
            0: "platform",
            1: "mouse",
            2: "spout",
            3: "pp_ball",
            4: "l_tube",
            5: "s_tube",
            6: "nest",
            7: "roach",
        }
    )
    return df


def get_area(rect):
    """
    Calculate area given four coordinates
    :param rect: array with rectangle coordinates
    """
    return (rect[2] - rect[0]) * (rect[3] - rect[1])


def get_corners(rect):
    """
    Get corners of rectangle given four coordinates
    :param rect: array with rectangle coordinates
    """
    rect_corners = []
    rect_corners.append(rect[0] - rect[2] * 0.5)  # x1
    rect_corners.append(rect[1] - rect[3] * 0.5)  # y1
    rect_corners.append(rect[0] + rect[2] * 0.5)  # x2
    rect_corners.append(rect[1] + rect[3] * 0.5)  # y2
    return rect_corners


def eval_overlap(R1, R2):
    """
    Determine overlap between two regions
    :param R1: first region
    :param R2: second region
    """
    if (R1[0] >= R2[2]) or (R1[2] <= R2[0]) or (R1[3] <= R2[1]) or (R1[1] >= R2[3]):
        return False
    else:
        return True


def measure_overlap(R1, R2):
    """
    Calculate overlap between two regions
    :param R1: first region
    :param R2: second region
    """
    dx = min([R1[2], R2[2]]) - max(R1[0], R2[0])
    dy = min([R1[3], R2[3]]) - min([R1[1], R2[1]])
    if (dx >= 0) and (dy >= 0):
        return dx * dy / 2
    else:
        return None


def get_xywh(R):
    """
    Get centroid and rectangle size information
    :param R: rectangle coordinates
    """
    x = np.mean([[R[0], R[2]]])
    y = np.mean([R[1], R[3]])
    w = R[2] - R[0]
    h = R[3] - R[1]
    return [x, y, w, h]


def group_rectangles(labels_df, eps=0.1):
    """
    Depending on overlap, merges two regions or removes lower confidence region
    :param labels: objects to merge and/or remove
    :param eps: parameter determine amount of overlap
    """
    highconf_rect_values = (
        labels_df.sort_values(by="confidence", ascending=False)
        .iloc[0, :][["x_center", "y_center", "x_width", "y_height"]]
        .values.tolist()
    )
    other_rect_values = (
        labels_df.sort_values(by="confidence", ascending=False)
        .iloc[1:, :][["x_center", "y_center", "x_width", "y_height"]]
        .values.tolist()
    )
    R1 = get_corners(highconf_rect_values)
    R_grouped = [R1]
    R1_area = get_area(R1)
    for other_rect_values_x in other_rect_values:
        R2 = get_corners(other_rect_values_x)
        R2_area = get_area(R2)
        is_overlap = eval_overlap(R1, R2)
        if is_overlap:
            overlap_area = measure_overlap(R1, R2)
            perc_overlap = overlap_area / min([R1_area, R2_area])
            if perc_overlap >= eps:
                R_grouped.append(R2)
    R_grouped = np.array(R_grouped)
    grouped_R = [
        np.min(R_grouped[:, 0]),
        np.min(R_grouped[:, 1]),
        np.max(R_grouped[:, 2]),
        np.max(R_grouped[:, 3]),
    ]
    grouped_xywh = get_xywh(grouped_R)
    label = labels_df.iloc[0]["label"]
    max_conf = np.max(labels_df["confidence"])
    grouped_labels_df = pd.DataFrame(
        columns=["label", "x_center", "y_center", "x_width", "y_height", "confidence"],
        data=[[label] + grouped_xywh + [max_conf]],
    )
    return grouped_labels_df


def run_process(
    animal,
    labels_to_be_grouped,
    labels_to_be_propagated,
    max_count_per_label,
    preds_path,
    out_path,
):
    """
    Primary function to run post-processing
    :param animal: name of animal
    :param labels_to_be_grouped: an array labels to group and/or remove if low confidence
    :param labels_to_be_propagated: an array labels to propagate to next frame
    :max_count_per_label: a dictionary with the maximum number of objects to be detected for a class (takes the most confident)
    :param preds_path: location to folder of pre-processed output
    :param out_path: location to store processed output
    """
    preds_filenames = [preds_path + fn for fn in os.listdir(preds_path)]
    preds_filenames
    preds_f_idxs = []
    for idx, preds_filename in enumerate(preds_filenames):
        if preds_filename.rsplit("_")[-1].rstrip(".txt") == "Store":
            continue
        preds_f_idx = int(preds_filename.rsplit("_")[-1].rstrip(".txt"))
        preds_f_idxs.append(preds_f_idx)
    preds_filenames = np.array(preds_filenames)[np.argsort(preds_f_idxs)].tolist()
    print("out_path ", out_path, flush=True)
    start = t.time()
    for idx, preds_filename in enumerate(preds_filenames):
        current_time = t.time()
        if preds_filename == ".DS_Store":
            continue
        preds_df = load_preds_file(preds_filename)
        for label in labels_to_be_grouped:
            labels_df = preds_df[preds_df["label"] == label]
            if len(labels_df) == 0:
                continue
            elif (
                len(labels_df) == 1
            ):  # if there is only one label then we basically don't have to group
                grouped_labels_df = labels_df
            elif len(labels_df) > 1:  # if it is greater than 1 then we group labels
                grouped_labels_df = group_rectangles(labels_df)
        for (
            class_,
            count,
        ) in (
            max_count_per_label.items()
        ):  # if there are more than "count" of a class detected, takes the "count" most confident
            other_labels_df = grouped_labels_df[grouped_labels_df["label"] != label]
            label_df = grouped_labels_df[grouped_labels_df["label"] != label]
            label_df = label_df.sort_values(by="confidence", ascending=False).head(
                count
            )
            grouped_labels_df = pd.concat([other_labels_df, label_df])
        preds_df = preds_df[preds_df["label"] != label]
        preds_df = pd.concat([preds_df, grouped_labels_df], ignore_index=True)
        preds_df.to_csv(out_path + preds_filename.rsplit("/")[-1], index=False)


if __name__ == "__main__":
    import os

    # print(os.environ['CONDA_DEFAULT_ENV'])
    print("[running]*100", flush=True)
    run_process(
        "",
        ["platform", "mouse", "spout", "l_tube", "s_tube", "roach"],
        [],
        {"pp_ball": 2},
        sys.argv[1],
        sys.argv[2],
    )
    print("[finished running]*100", flush=True)
