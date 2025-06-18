
import argparse
import numpy as np

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.taplib import TensorTiler2D, TensorAccessSequence

import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.utils.trace_events_enum import CoreEvent, ShimTileEvent, MemTileEvent

dtype_map = {
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
}

def main(opts):
    with mlir_mod_ctx() as ctx:
        microbenchmark(
            opts.dev,
            opts.dtype_in,
            opts.dtype_out,
            opts.input_netlist,
            opts.trace_size,
        )
        print(ctx.module)

def parse_netlist(input_netlist):
    pass

def microbenchmark(
    dev,
    dtype_in_str,
    dtype_out_str,
    input_netlist,
    trace_size,
):
    pass


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="Microbenchmark",
        description="Emits MLIR code for microbenchmark of the given netlist topology.",
    )
    argparser.add_argument(
        "--dev", 
        type=str, 
        dest="dev",
        default="npu2",
    )
    argparser.add_argument(
        "-nl",
        "--netlist", 
        type=str, 
        dest="netlist",
        default="netlist.json",
    )
    argparser.add_argument(
        "--dtype", 
        type=str, 
        dest="dtype_str",
        choices=["i8", "i16", "i32"], 
        default="i32"
    )
    argparser.add_argument(
        "--trace_size", 
        type=int, 
        default=0
    )
    opts = argparser.parse_args()
    main(opts)