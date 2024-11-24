import re
import matplotlib.pyplot as plt
import numpy as np
import json

# TODO (Ron Wang): Refactor this into a class


def read_events_from_log(logging_file):
    """Create chrome tracing-formatted events from logging_file. Can be refactored to support more stages."""

    read_pattern = r".*pid=(\d+).*\[Binary Wall Time\] (\d*\.\d*) (\d*\.\d*) (\d*).*"
    preprocess_pattern = (
        r".*pid=(\d+).*Preprocess Wall Time\] (\d*\.\d*) (\d*\.\d*) (\d*).*"
    )
    inference_pattern = (
        r"\[MapBatches\(Classifier\) Wall Time\] (\d*\.\d*) (\d*\.\d*) (\d*).*"
    )

    read_timestamps, read_wall_times, read_num_rows = [], [], []
    preprocess_timestamps, preprocess_wall_times, preprocess_num_rows = [], [], []
    inference_timestamps, inference_wall_times, inference_num_rows = [], [], []

    events = []
    with open(logging_file, "r") as f:
        for line in f:
            match_read = re.search(read_pattern, line)
            match_preprocess = re.search(preprocess_pattern, line)
            match_inference = re.search(inference_pattern, line)
            if match_read:
                pid = match_read.group(1)
                timestamp = float(match_read.group(2))
                duration = float(match_read.group(3))
                num_rows = float(match_read.group(4))

                read_timestamps.append(timestamp)
                read_wall_times.append(duration)
                read_num_rows.append(num_rows)
                event = {
                    "name": "Read",
                    "ph": "X",
                    "ts": timestamp * 1e6,
                    "dur": duration * 1e6,
                    "pid": pid,
                    "tid": "Read",
                    "cname": "rail_load",
                    "args": {},
                }
                events.append(event)
            elif match_preprocess:
                pid = match_preprocess.group(1)
                timestamp = float(match_preprocess.group(2))
                duration = float(match_preprocess.group(3))
                num_rows = float(match_preprocess.group(4))

                preprocess_timestamps.append(timestamp)
                preprocess_wall_times.append(duration)
                preprocess_num_rows.append(num_rows)
                event = {
                    "name": "Preprocess",
                    "ph": "X",
                    "ts": timestamp * 1e6,
                    "dur": duration * 1e6,
                    "pid": pid,
                    "tid": "Preprocess",
                    "cname": "rail_idle",
                    "args": {},
                }
                events.append(event)
            elif match_inference:
                timestamp = float(match_inference.group(1))
                duration = float(match_inference.group(2))
                num_rows = float(match_inference.group(3))

                inference_timestamps.append(timestamp)
                inference_wall_times.append(duration)
                inference_num_rows.append(num_rows)
                event = {
                    "name": "Inference",
                    "ph": "X",
                    "ts": timestamp * 1e6,
                    "dur": duration * 1e6,
                    "pid": "GPU",
                    "tid": "Inference",
                    "cname": "rail_animation",
                    "args": {},
                }
                events.append(event)

    results = {
        "read_timestamps": read_timestamps,
        "read_wall_times": read_wall_times,
        "read_num_rows": read_num_rows,
        "preprocess_timestamps": preprocess_timestamps,
        "preprocess_wall_times": preprocess_wall_times,
        "preprocess_num_rows": preprocess_num_rows,
        "inference_timestamps": inference_timestamps,
        "inference_wall_times": inference_wall_times,
        "inference_num_rows": inference_num_rows,
    }
    print(
        "Found {} read events, {} preprocess events, and {} inference events.".format(
            len(read_timestamps), len(preprocess_timestamps), len(inference_timestamps)
        )
    )
    return results, events


def get_mean_tput(results, stage_name: str):
    """Computes mean tput of a single Ray Task/Actor"""
    wall_times = results[f"{stage_name}_wall_times"]
    num_rows = results[f"{stage_name}_num_rows"]

    print(
        "Mean tput for {} stage: {}".format(
            stage_name, np.sum(num_rows) / np.sum(wall_times)
        )
    )
    return np.sum(num_rows) / np.sum(wall_times)


def get_median_tput(results, stage_name: str):
    """Computes median tput of a single Ray Task/Actor"""
    wall_times = np.array(results[f"{stage_name}_wall_times"])
    num_rows = np.array(results[f"{stage_name}_num_rows"])

    print(
        "Median tput for {} stage: {}".format(
            stage_name, np.median(num_rows / wall_times)
        )
    )
    return np.median(num_rows / wall_times)


def get_events_from_name(events, stage_name: str):
    return [event for event in events if event["name"] == stage_name]


def get_num_pids(events, stage_name: str):
    """Get the number of pids/Ray workers launched for events, where events could be from a specific stage."""
    pids = set()
    for event in events:
        pids.add(event["pid"])
    print("Number of pids for {} stage: {}".format(stage_name, len(pids)))
    return pids, len(pids)


def compute_data_stall(events):
    stall_times = []
    timestamps = []
    events.sort(key=lambda x: x["ts"])
    for i in range(len(events) - 1):
        current_event = events[i]
        next_event = events[i + 1]

        stall_time = max(
            next_event["ts"] - (current_event["ts"] + current_event["dur"]), 0
        )
        stall_times.append(stall_time / 1e6)
        timestamps.append(next_event["ts"])
    return stall_times, timestamps


def get_mean_data_stall(events, stage_name: str):
    data_stall, _ = compute_data_stall(events)
    print(
        "Mean data stall time for {} stage: {}".format(stage_name, np.mean(data_stall))
    )
    return np.mean(data_stall)


def plot_data_stall(events):
    read_events = get_events_from_name(events, "Read")
    preprocess_events = get_events_from_name(events, "Preprocess")
    inference_events = get_events_from_name(events, "Inference")

    read_data_stall, read_timestamps = compute_data_stall(read_events)
    preprocess_data_stall, preprocess_timestamps = compute_data_stall(preprocess_events)
    inference_data_stall, inference_timestamps = compute_data_stall(inference_events)

    plt.figure(figsize=(12, 6))
    plt.title("4 CPUs, Reading from Local")
    plt.ylim(0, 6)
    plt.scatter(read_timestamps, read_data_stall, label="Read Data Stall Time")
    plt.scatter(
        preprocess_timestamps, preprocess_data_stall, label="Preprocess Data Stall Time"
    )
    plt.scatter(
        inference_timestamps, inference_data_stall, label="Inference Data Stall Time"
    )

    plt.axhline(
        y=np.mean(read_data_stall),
        color="r",
        linestyle="--",
        label="Mean Read Data Stall Time",
    )
    plt.axhline(
        y=np.mean(preprocess_data_stall),
        color="g",
        linestyle="--",
        label="Mean Preprocess Data Stall Time",
    )
    plt.axhline(
        y=np.mean(inference_data_stall),
        color="b",
        linestyle="--",
        label="Mean Inference Data Stall Time",
    )
    plt.grid(True)
    plt.legend()
    plt.show()


def to_json(events, file_name: str = "temp.json"):
    json_output = json.dumps(events, indent=4)
    with open(file_name, "w") as file:
        file.write(json_output)


if __name__ == "__main__":
    results, events = read_events_from_log("cpu_4_s3_threads_0.txt")
    read_events = get_events_from_name(events, "Read")
    preprocess_events = get_events_from_name(events, "Preprocess")
    inference_events = get_events_from_name(events, "Inference")

    to_json(events, "cpu_4_s3_threads_0.json")

    read_num_workers = get_num_pids(read_events, "Read")[1]
    preprocess_num_workers = get_num_pids(preprocess_events, "Preprocess")[1]
    inference_num_workers = get_num_pids(inference_events, "Inference")[1]

    print("")
    read_tput = get_mean_tput(results, "read")
    preprocess_tput = get_mean_tput(results, "preprocess")
    inference_tput = get_mean_tput(results, "inference")

    print("")
    print("Concurrent Read Tput: {}".format(read_tput * read_num_workers))
    print(
        "Concurrent Preprocess Tput: {}".format(
            preprocess_tput * preprocess_num_workers
        )
    )
    print(
        "Concurrent Inference Tput: {}".format(inference_tput * inference_num_workers)
    )

    read_data_stall = get_mean_data_stall(read_events, "Read")
    preprocess_data_stall = get_mean_data_stall(preprocess_events, "Preprocess")
    inference_data_stall = get_mean_data_stall(inference_events, "Inference")

    print("")
    print(
        "Preprocess Data Stall / Read Data Stall: {}".format(
            np.mean(preprocess_data_stall) / np.mean(read_data_stall)
        )
    )
    print(
        "Inference Data Stall / Read Data Stall: {}".format(
            np.mean(inference_data_stall) / np.mean(read_data_stall)
        )
    )

    plot_data_stall(events)
