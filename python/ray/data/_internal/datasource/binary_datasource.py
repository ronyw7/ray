from typing import TYPE_CHECKING
import time

from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data.datasource.file_based_datasource import FileBasedDatasource

# @ronyw
from ray.data._internal.execution.prometheus_monitoring_service import record_metrics

if TYPE_CHECKING:
    import pyarrow


class BinaryDatasource(FileBasedDatasource):
    """Binary datasource, for reading and writing binary files."""

    _COLUMN_NAME = "bytes"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _read_stream(self, f: "pyarrow.NativeFile", path: str):
        start_time = time.perf_counter()
        data = f.readall()
        end_time = time.perf_counter()
        print(
            f"[{self.get_name()} Wall Time]",
            end_time,  # Timestamp
            (end_time - start_time),  # Wall Time
            self._rows_per_file(),  # Num Rows
            flush=True,
        )
        record_metrics("ReadBinary", self._rows_per_file(), end_time - start_time)

        builder = ArrowBlockBuilder()
        item = {self._COLUMN_NAME: data}
        builder.add(item)
        yield builder.build()

    def _rows_per_file(self):
        return 1
