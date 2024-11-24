from dataclasses import dataclass
from typing import List


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
    resource: str
    additional_num_pids: int



class PipelineBottleneckAnalyzer:
    def __init__(self, metrics: List[PipelineStageMetrics]):
        self.metrics = metrics

    def analyze(self):
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

        return ScalingRequest("CPU", num_pid_requested)


stage_read = PipelineStageMetrics(
    name="read",
    num_pids=2,
    mean_tput=1370.9166324818023,
    concurrent_tput=2741.8332649636045,
    overall_pipeline_tput=15.299708420701588,
    mean_data_stall=0.05670156308031243,
)

stage_preprocess = PipelineStageMetrics(
    name="preprocess",
    num_pids=2,
    mean_tput=9.333623736539899,
    concurrent_tput=18.667247473079797,
    overall_pipeline_tput=15.299708420701588,
    mean_data_stall=0.006802485588363757,
)

stage_inference = PipelineStageMetrics(
    name="inference",
    num_pids=1,
    mean_tput=32.35092887745521,
    concurrent_tput=32.35092887745521,
    overall_pipeline_tput=15.299708420701588,
    mean_data_stall=0.49926610687180073,
)

analyzer = PipelineBottleneckAnalyzer([stage_read, stage_preprocess, stage_inference])
request = analyzer.analyze()
print(request)
