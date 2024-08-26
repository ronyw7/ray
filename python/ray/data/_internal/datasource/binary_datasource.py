from typing import TYPE_CHECKING, List, Optional
import numpy as np
import time

from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data.datasource.file_based_datasource import FileBasedDatasource

if TYPE_CHECKING:
    import pyarrow


class BinaryDatasource(FileBasedDatasource):
    """Binary datasource, for reading and writing binary files."""

    _COLUMN_NAME = "bytes"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._last_output_time = None

    def _read_stream(self, f: "pyarrow.NativeFile", path: str):
        start_time = time.perf_counter()
        # if self._last_output_time:
        #     print(
        #         f"[{self.get_name()} Data Stall Time]",
        #         start_time,  # Timestamp
        #         (start_time - self._last_output_time),  # Data Stall Time
        #         flush=True,
        #     )

        data = f.readall()
        end_time = time.perf_counter()
        print(
            f"[{self.get_name()} Wall Time]",
            end_time,  # Timestamp
            (end_time - start_time),  # Wall Time
            self._rows_per_file(),  # Num Rows
            flush=True,
        )

        # self._last_output_time = end_time

        builder = ArrowBlockBuilder()
        item = {self._COLUMN_NAME: data}
        builder.add(item)
        yield builder.build()

    def _rows_per_file(self):
        return 1
