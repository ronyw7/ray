import ray
import time
import numpy as np
import subprocess

from typing import List
from dataclasses import dataclass

from ray.data._internal.execution.system_metrics_logger import SystemMetricsLogger
from ray.util.state import list_nodes


@ray.remote(num_cpus=0)
class PipelineMetricsTracker:
    def __init__(self):
        # Pipeline metrics
        self.raw_metrics = {}
        self.stall_metrics = {}
        self.summary_metrics = {}

        # System metrics
        self.system_metrics = []
        self.system_metrics_logger = SystemMetricsLogger(interval=5)

        # Data pipeline variables
        self.pipeline_first_output_time = None
        self.sink_stage_name = "inference"

        # Analysis
        self.analyzer = PipelineBottleneckAnalyzer()
        self.should_record_metric = True
        self.last_analysis_time = time.perf_counter()

        self.scaling_request = []

    def record(
        self, stage: str, pid: str, end_time: float, wall_time: float, num_rows: int
    ):
        if not self.should_record_metric:
            return

        # Collect pipeline metrics
        if not self.raw_metrics.get(stage):
            self.raw_metrics[stage] = []
        self.raw_metrics[stage].append(
            {"pid": pid, "time": end_time, "wall_time": wall_time, "rows": num_rows}
        )
        if not self.pipeline_first_output_time:
            self.pipeline_first_output_time = end_time

        # Collect system metrics
        if self.system_metrics_logger.should_collect():
            self.system_metrics.append(self.system_metrics_logger.log_once()[0])

        # Analysis and scaling logic
        if self.analyzer.should_analyze():
            # Generate summary metrics
            self.compute_summary_metrics()
            self.analyzer.add_metrics(self.output_summary_metrics())

            request = self.analyzer.analyze()
            if request and not self.scaling_request:
                self.scaling_request.append(request)
                print(f"Generated scaling request: {request}")

        if self.scaling_request:
            self.should_record_metric = False
            scaling_request = self.scaling_request[0]
            self.scaling_request.clear()
            scaler = PipelineResourceScaler(scaling_request)
            scaler.scale()

        self.should_record_metric = True

    def compute_overall_tput(self):
        """Computes the overall tput for the pipeline."""
        end_time = self.raw_metrics[self.sink_stage_name][-1]["time"]
        num_rows = np.sum([r["rows"] for r in self.raw_metrics[self.sink_stage_name]])
        return num_rows / (end_time - self.pipeline_first_output_time)

    def compute_data_stall(self):
        for stage, events in self.raw_metrics.copy().items():
            stall_times = []

            for i in range(len(events) - 1):
                current_event = events[i]
                next_event = events[i + 1]

                stall_time = max(
                    next_event["time"]
                    - (current_event["time"] + current_event["wall_time"]),
                    0,
                )
                stall_times.append(stall_time)
            self.stall_metrics[stage] = stall_times

    def compute_summary_metrics(self):
        self.compute_data_stall()
        for stage, events in self.raw_metrics.copy().items():
            self.summary_metrics[stage] = {
                "total_wall_time": np.sum([r["wall_time"] for r in events]),
                "total_num_rows": np.sum([r["rows"] for r in events]),
            }
            # num pids
            self.summary_metrics[stage]["num_pids"] = len(
                set([r["pid"] for r in events])
            )
            # mean tput
            self.summary_metrics[stage]["mean_tput"] = (
                self.summary_metrics[stage]["total_num_rows"]
                / self.summary_metrics[stage]["total_wall_time"]
            )
            # concurrent tput
            self.summary_metrics[stage]["concurrent_tput"] = (
                self.summary_metrics[stage]["mean_tput"]
                * self.summary_metrics[stage]["num_pids"]
            )

        for stage, events in self.stall_metrics.items():
            # mean data stall
            self.summary_metrics[stage]["mean_data_stall"] = np.mean(events)

        self.overall_tput = self.compute_overall_tput()

    def output_summary_metrics(self):
        """Generate PipelineStageMetrics objects for each stage in the pipeline."""
        stage_metrics = []
        for stage, metrics in self.summary_metrics.items():
            stage_metrics.append(
                PipelineStageMetrics(
                    name=stage,
                    num_pids=metrics["num_pids"],
                    mean_tput=metrics["mean_tput"],
                    concurrent_tput=metrics["concurrent_tput"],
                    overall_pipeline_tput=self.overall_tput,
                    mean_data_stall=metrics["mean_data_stall"],
                )
            )
        return stage_metrics

    def find_bottleneck_stage(self):
        self.compute_summary_metrics()
        bottleneck_stage = None
        lowest_tput = float("inf")

        for stage, metrics in self.summary_metrics.items():
            if metrics["mean_tput"] < lowest_tput:
                lowest_tput = metrics["mean_tput"]
                bottleneck_stage = stage

        return bottleneck_stage, lowest_tput

    def print_summary(self):
        self.compute_summary_metrics()
        for stage, metrics in self.summary_metrics.items():
            print(f"Stage: {stage}")
            print(f"Total Wall Time: {metrics['total_wall_time']}")
            print(f"Total Num Rows: {metrics['total_num_rows']}")
            print(f"Num Pids: {metrics['num_pids']}")
            print(f"Mean Tput: {metrics['mean_tput']}")
            print(f"Concurrent Tput: {metrics['concurrent_tput']}")
            print(f"Mean Data Stall: {metrics['mean_data_stall']}")

            print(f"Overall Pipeline Tput: {self.overall_tput}")
            print("")
        print(self.system_metrics)


@dataclass
class PipelineStageMetrics:
    """
    e.g.
    stage: read
    mean_tput: 1000
    num_pids: 2
    concurrent_tput: 2000
    data_stall: 0.001
    """

    name: str
    num_pids: int
    mean_tput: float
    concurrent_tput: float
    overall_pipeline_tput: float
    mean_data_stall: float


@dataclass
class ScalingRequest:
    """
    e.g.
    resource: CPU
    additional_num_pids: 2
    """

    resource: str
    additional_num_pids: int


class PipelineBottleneckAnalyzer:
    def __init__(self, interval=20):
        self.interval = interval
        self.last_analysis_time = time.perf_counter()

    def add_metrics(self, metrics: List[PipelineStageMetrics]):
        self.metrics = metrics

    def should_analyze(self):
        return time.perf_counter() - self.last_analysis_time >= self.interval

    def analyze(self):
        # For now, we only consider the CPU-bound cases; to add GPU request later.
        bottleneck_stage = None
        lowest_tput = float("inf")

        inference_tput = None

        for stage in self.metrics:
            if stage.concurrent_tput < lowest_tput:
                lowest_tput = stage.concurrent_tput
                bottleneck_stage = stage
            if stage.name == "inference":
                inference_tput = stage.concurrent_tput

        num_pid_optim = inference_tput // bottleneck_stage.mean_tput + 1
        num_pid_requested = num_pid_optim - bottleneck_stage.num_pids
        print("Bottleneck stage:", bottleneck_stage, "Throughput:", lowest_tput)

        self.last_analysis_time = time.perf_counter()
        return ScalingRequest("CPU", num_pid_requested)


class PipelineResourceScaler:
    def __init__(self, request: ScalingRequest):
        self.request = request

    def scale(self):
        print(f"Received scaling request: {self.request}")
        print(list_nodes())

        additional_num_pids = int(self.request.additional_num_pids)
        # ray.init() does not expose necessary ports for other workers to join
        # To fix, run `ray start --head` in the terminal and then ray.init("auto") in the script
        ray_start_command = f"ray start --address=10.3.0.4:6379 --num-cpus {additional_num_pids} --num-gpus 0"

        try:
            subprocess.run(ray_start_command, shell=True, check=True)
            print(
                f"Successfully started {additional_num_pids} workers. Run `ray status` to check the current cluster status."
            )

            ray_status_command = "ray status"
            result = subprocess.run(
                ray_status_command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )

            print("\nCluster status after scaling:\n")
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"Failed to start {additional_num_pids} workers. Error: {e}")
