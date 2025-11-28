import os
import sys
import glob
import argparse
import csv
import yaml
import json
import re
import math
import copy
from collections import OrderedDict
from openpyxl import Workbook
from openpyxl.chart import ScatterChart, Reference, Series
from openpyxl.chart.axis import Scaling
from openpyxl.chart.text import RichText
from openpyxl.chart.layout import Layout, ManualLayout
from openpyxl.drawing.text import Paragraph, ParagraphProperties, CharacterProperties, Font   

# This class represents a single field in the test result
class ResultField:
    def __init__(self, name: str, description: str, default_value="N/A", value_unit="", hidden_field=False):
        self.name = name
        self.description = description
        self.value = default_value
        self.unit = value_unit
        self.hidden_field = hidden_field

    def __repr__(self):
        return f"{self.description}: {self.value} {self.unit}"


# This class represents the test result for a single benchmark task
class TestResult:
    def __init__(self, result_dir: str, benchmark_name: str, task_name: str):
        self.result_dir = result_dir
        self.benchmark_name = benchmark_name
        self.task_name = task_name
        self.fields = OrderedDict()
        self.fields = {
            "benchmark_name": ResultField("benchmark_name", "Name of the benchmark", default_value=self.benchmark_name),
            "task_name": ResultField("task_name", "Name of the task to be tested, normaly corresponds to a specific configuration of the benchmark", default_value=self.task_name),
            "num_compute_tile": ResultField("num_compute_tile", "Number of compute tiles used in the task"),
            "num_mem_tile": ResultField("num_mem_tile", "Number of memory tiles used in the task"),
            "num_shim_tile": ResultField("num_shim_tile", "Number of shim tiles used in the task"),
            "num_objectFIFO": ResultField("num_objectFIFO", "Number of object FIFO used in the task"),
            "num_unicast_objectFIFO": ResultField("num_unicast_objectFIFO", "Number of object FIFO that is doing unicast in the task"),
            "num_multicast_objectFIFO": ResultField("num_multicast_objectFIFO", "Number of object FIFO that is doing multicast in the task"),
            "num_objectFIFO_link": ResultField("num_objectFIFO_link", "Number of object FIFO link used in the task"),
            "build_time [s]": ResultField("build_time [s]", "Time taken for building the mlir file from source code (This includes PnR time if PnR is invoked during build)", value_unit="seconds"),
            "pnr_time [s]": ResultField("pnr_time [s]", "Time taken for placing and routing the design", value_unit="seconds"),
            "compilation_time [s]": ResultField("compilation_time [s]", "Time taken for compiling the design to binary from the mlir file", value_unit="seconds"),
            "total_end2end_compilation_time [s]": ResultField("total_end2end_compilation_time [s]", "Total time taken for end-to-end compilation from source code to binary", value_unit="seconds"),
            "avg_NPU_runtime [us]": ResultField("avg_NPU_runtime [us]", "Average runtime of the final placed and routed design on the NPU", value_unit="microseconds"),
            "num_SA_moves": ResultField("num_SA_moves", "Number of moves performed during Simulated Annealing placement"),
            "num_SA_legal_moves": ResultField("num_SA_legal_moves", "Number of legal moves during Simulated Annealing placement"),
            "SA_legal_move_rate [%]": ResultField("SA_legal_move_rate [%]", "Rate of legal moves during Simulated Annealing placement", value_unit="%"),
            "num_SA_accepted_legal_moves": ResultField("num_SA_accepted_legal_moves", "Number of accepted legal moves during Simulated Annealing placement"),
            "SA_legal_move_acceptance_rate [%]": ResultField("SA_legal_move_acceptance_rate [%]", "Acceptance rate of legal moves during Simulated Annealing placement", value_unit="%"),
            "num_SA_rejected_legal_moves": ResultField("num_SA_rejected_legal_moves", "Number of rejected legal moves during Simulated Annealing placement"),
            "SA_legal_move_rejection_rate [%]": ResultField("SA_legal_move_rejection_rate [%]", "Rejection rate of legal moves during Simulated Annealing placement", value_unit="%"),
            "num_SA_congested_moves": ResultField("num_SA_congested_moves", "Number of moves that is illegal because it cannout find a legal routing during Simulated Annealing placement"),
            "SA_congested_move_rate [%]": ResultField("SA_congested_move_rate [%]", "Rate of congested moves during Simulated Annealing placement", value_unit="%"),
            "num_SA_accepted_congested_moves": ResultField("num_SA_accepted_congested_moves", "Number of accepted congested moves during Simulated Annealing placement"),
            "SA_congested_move_acceptance_rate [%]": ResultField("SA_congested_move_acceptance_rate [%]", "Acceptance rate of congested moves during Simulated Annealing placement", value_unit="%"),
            "num_SA_rejected_congested_moves": ResultField("num_SA_rejected_congested_moves", "Number of rejected congested moves during Simulated Annealing placement"),
            "SA_congested_move_rejection_rate [%]": ResultField("SA_congested_move_rejection_rate [%]", "Rejection rate of congested moves during Simulated Annealing placement", value_unit="%"),
            "num_SA_illegal_moves": ResultField("num_SA_illegal_moves", "Number of illegal moves during Simulated Annealing placement"),
            "SA_illegal_move_rate [%]": ResultField("SA_illegal_move_rate [%]", "Rate of illegal moves during Simulated Annealing placement", value_unit="%"),
            "num_SA_accepted_illegal_moves": ResultField("num_SA_accepted_illegal_moves", "Number of accepted illegal moves during Simulated Annealing placement"),
            "SA_illegal_move_acceptance_rate [%]": ResultField("SA_illegal_move_acceptance_rate [%]", "Acceptance rate of illegal moves during Simulated Annealing placement", value_unit="%"),
            "num_SA_rejected_illegal_moves": ResultField("num_SA_rejected_illegal_moves", "Number of rejected illegal moves during Simulated Annealing placement"),
            "SA_illegal_move_rejection_rate [%]": ResultField("SA_illegal_move_rejection_rate [%]", "Rejection rate of illegal moves during Simulated Annealing placement", value_unit="%"),
            "num_SA_accepted_moves": ResultField("num_SA_accepted_moves", "Number of accepted moves during Simulated Annealing placement"),
            "SA_move_acceptance_rate [%]": ResultField("SA_move_acceptance_rate [%]", "Acceptance rate of moves during Simulated Annealing placement", value_unit="%"),
            "num_SA_legal_accepted_moves": ResultField("num_SA_legal_accepted_moves", "Number of accepted legal moves during Simulated Annealing placement"),
            "SA_accepted_move_legal_rate [%]": ResultField("SA_accepted_move_legal_rate [%]", "The percentage of accepted moves that are legal during Simulated Annealing placement", value_unit="%"),
            "num_SA_congested_accepted_moves": ResultField("num_SA_congested_accepted_moves", "Number of accepted congested moves during Simulated Annealing placement"),
            "SA_accepted_move_congested_rate [%]": ResultField("SA_accepted_move_congested_rate [%]", "The percentage of accepted moves that are congested during Simulated Annealing placement", value_unit="%"),
            "num_SA_illegal_accepted_moves": ResultField("num_SA_illegal_accepted_moves", "Number of accepted illegal moves during Simulated Annealing placement"),
            "SA_accepted_move_illegal_rate [%]": ResultField("SA_accepted_move_illegal_rate [%]", "The percentage of accepted moves that are illegal during Simulated Annealing placement", value_unit="%"),
            "num_SA_rejected_moves": ResultField("num_SA_rejected_moves", "Number of rejected moves during Simulated Annealing placement"),
            "SA_move_rejection_rate [%]": ResultField("SA_move_rejection_rate [%]", "Rejection rate of moves during Simulated Annealing placement", value_unit="%"),
            "num_SA_legal_rejected_moves": ResultField("num_SA_legal_rejected_moves", "Number of rejected legal moves during Simulated Annealing placement"),
            "SA_rejected_move_legal_rate [%]": ResultField("SA_rejected_move_legal_rate [%]", "The percentage of rejected moves that are legal during Simulated Annealing placement", value_unit="%"),
            "num_SA_congested_rejected_moves": ResultField("num_SA_congested_rejected_moves", "Number of rejected congested moves during Simulated Annealing placement"),
            "SA_rejected_move_congested_rate [%]": ResultField("SA_rejected_move_congested_rate [%]", "The percentage of rejected moves that are congested during Simulated Annealing placement", value_unit="%"),
            "num_SA_illegal_rejected_moves": ResultField("num_SA_illegal_rejected_moves", "Number of rejected illegal moves during Simulated Annealing placement"),
            "SA_rejected_move_illegal_rate [%]": ResultField("SA_rejected_move_illegal_rate [%]", "The percentage of rejected moves that are illegal during Simulated Annealing placement", value_unit="%"),
            "SA_per_move_info": ResultField("SA_per_move_info", "Detailed per-move information during Simulated Annealing placement", default_value=[], hidden_field=True),
            "num_neighbour_sharing_objectFIFO": ResultField("num_neighbour_sharing_objectFIFO", "Number of object FIFO connections that are using neighbour sharing routing in the final placement"),
            "num_circuit_switch_objectFIFO": ResultField("num_circuit_switch_objectFIFO", "Number of object FIFO connections that are using circuit switch routing in the final placement"),
            "num_packet_flow_objectFIFO": ResultField("num_packet_flow_objectFIFO", "Number of object FIFO connections that are using packet flow in the final placement"),
            "total_num_circuit_switch_tracks": ResultField("total_num_circuit_switch_tracks", "Total number of circuit switch tracks used in the final placement"),
            "largest_circuit_switch_net": ResultField("largest_circuit_switch_net", "Largest number of tracks used by a single circuit switch net in the final placement"),
            "smallest_circuit_switch_net": ResultField("smallest_circuit_switch_net", "Smallest number of tracks used by a single circuit switch net in the final placement"),
            "avg_track_per_circuit_switch_net": ResultField("avg_track_per_circuit_switch_net", "Average number of tracks used per circuit switch net in the final placement"),
            "longest_circuit_switch_path": ResultField("longest_circuit_switch_path", "Longest path (in hops) of a circuit switch net in the final placement"),
            "shortest_circuit_switch_path": ResultField("shortest_circuit_switch_path", "Shortest path (in hops) of a circuit switch net in the final placement"),
            "total_num_packet_flow_tracks": ResultField("total_num_packet_flow_tracks", "Total number of packet flow tracks used in the final placement"),
            "largest_packet_flow_net": ResultField("largest_packet_flow_net", "Largest number of tracks used by a single packet flow net in the final placement"),
            "smallest_packet_flow_net": ResultField("smallest_packet_flow_net", "Smallest number of tracks used by a single packet flow net in the final placement"),
            "avg_track_per_packet_flow_net": ResultField("avg_track_per_packet_flow_net", "Average number of tracks used per packet flow net in the final placement"),
            "longest_packet_flow_path": ResultField("longest_packet_flow_path", "Longest path (in hops) of a packet flow net in the final placement"),
            "shortest_packet_flow_path": ResultField("shortest_packet_flow_path", "Shortest path (in hops) of a packet flow net in the final placement"),
            "total_num_dma_ports": ResultField("total_num_dma_ports", "Total number of DMA ports used in the final placement"),
            "total_num_dma_in_ports": ResultField("total_num_dma_in_ports", "Total number of DMA input ports used in the final placement"),
            "total_num_dma_out_ports": ResultField("total_num_dma_out_ports", "Total number of DMA output ports used in the final placement"),
            "total_buffer_size [bytes]": ResultField("total_buffer_size [bytes]", "Total buffer size (in bytes) used in the final placement", value_unit="bytes"),
            "total_buffer_size_on_mem [bytes]": ResultField("total_buffer_size_on_mem [bytes]", "Total buffer size (in bytes) allocated on memory tiles in the final placement", value_unit="bytes"),
            "avg_buffer_size_on_mem [bytes]": ResultField("avg_buffer_size_on_mem [bytes]", "Average buffer size (in bytes) allocated on memory tiles in the final placement", value_unit="bytes"),
            "total_buffer_size_on_compute [bytes]": ResultField("total_buffer_size_on_compute [bytes]", "Total buffer size (in bytes) allocated on compute tiles in the final placement", value_unit="bytes"),
            "avg_buffer_size_on_compute [bytes]": ResultField("avg_buffer_size_on_compute [bytes]", "Average buffer size (in bytes) allocated on compute tiles in the final placement", value_unit="bytes"),
        }

        # Load the result values from the result directory
        if os.path.exists(result_dir):
            # Parse build stage results
            self._parse_build_stage_results()

            # Parse PnR Stage results
            self._parse_pnr_stage_results()

            # Parse AIECC compilation stage results
            self._parse_aiecc_compilation_stage_results()

            # Parse NPU run stage results
            self._parse_npu_run_stage_results()

    def __repr__(self):
        # prefix components:
        prt_space =  '    '
        prt_branch = '│   '
        # pointers:
        prt_tee =    '├── '
        prt_last =   '└── '
        result = f"TestResult: {self.benchmark_name} - {self.task_name}\n"
        result+= f"{prt_space}{self.fields["num_compute_tile"]}\n"
        result+= f"{prt_space}{self.fields["num_mem_tile"]}\n"
        result+= f"{prt_space}{self.fields["num_shim_tile"]}\n"
        result+= f"{prt_space}{self.fields["num_objectFIFO"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["num_unicast_objectFIFO"]}\n"
        result+= f"{prt_space}{prt_last}{self.fields["num_multicast_objectFIFO"]}\n"
        result+= f"{prt_space}{self.fields["num_objectFIFO_link"]}\n"
        result+= f"{prt_space}{self.fields["total_end2end_compilation_time [s]"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["build_time [s]"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_last}{self.fields["pnr_time [s]"]}\n"
        result+= f"{prt_space}{prt_last}{self.fields["compilation_time [s]"]}\n"
        result+= f"{prt_space}{self.fields["avg_NPU_runtime [us]"]}\n"
        result+= f"{prt_space}{self.fields["num_SA_moves"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["num_SA_legal_moves"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["SA_legal_move_rate [%]"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["num_SA_accepted_legal_moves"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["SA_legal_move_acceptance_rate [%]"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["num_SA_rejected_legal_moves"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_last}{self.fields["SA_legal_move_rejection_rate [%]"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["num_SA_congested_moves"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["SA_congested_move_rate [%]"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["num_SA_accepted_congested_moves"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["SA_congested_move_acceptance_rate [%]"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["num_SA_rejected_congested_moves"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_last}{self.fields["SA_congested_move_rejection_rate [%]"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["num_SA_illegal_moves"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["SA_illegal_move_rate [%]"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["num_SA_accepted_illegal_moves"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["SA_illegal_move_acceptance_rate [%]"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["num_SA_rejected_illegal_moves"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_last}{self.fields["SA_illegal_move_rejection_rate [%]"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["num_SA_accepted_moves"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["SA_move_acceptance_rate [%]"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["num_SA_legal_accepted_moves"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["SA_accepted_move_legal_rate [%]"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["num_SA_congested_accepted_moves"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["SA_accepted_move_congested_rate [%]"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["num_SA_illegal_accepted_moves"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_last}{self.fields["SA_accepted_move_illegal_rate [%]"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["num_SA_rejected_moves"]}\n"
        result+= f"{prt_space}{prt_last}{self.fields["SA_move_rejection_rate [%]"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["num_SA_legal_rejected_moves"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["SA_rejected_move_legal_rate [%]"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["num_SA_congested_rejected_moves"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["SA_rejected_move_congested_rate [%]"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_tee}{self.fields["num_SA_illegal_rejected_moves"]}\n"
        result+= f"{prt_space}{prt_branch}{prt_last}{self.fields["SA_rejected_move_illegal_rate [%]"]}\n"
        result+= f"{prt_space}{self.fields["num_neighbour_sharing_objectFIFO"]}\n"
        result+= f"{prt_space}{self.fields["num_circuit_switch_objectFIFO"]}\n"
        result+= f"{prt_space}{self.fields["num_packet_flow_objectFIFO"]}\n"
        result+= f"{prt_space}{self.fields["total_num_circuit_switch_tracks"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["largest_circuit_switch_net"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["smallest_circuit_switch_net"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["avg_track_per_circuit_switch_net"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["longest_circuit_switch_path"]}\n"
        result+= f"{prt_space}{prt_last}{self.fields["shortest_circuit_switch_path"]}\n"
        result+= f"{prt_space}{self.fields["total_num_packet_flow_tracks"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["largest_packet_flow_net"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["smallest_packet_flow_net"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["avg_track_per_packet_flow_net"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["longest_packet_flow_path"]}\n"
        result+= f"{prt_space}{prt_last}{self.fields["shortest_packet_flow_path"]}\n"
        result+= f"{prt_space}{self.fields["total_num_dma_ports"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["total_num_dma_in_ports"]}\n"
        result+= f"{prt_space}{prt_last}{self.fields["total_num_dma_out_ports"]}\n"
        result+= f"{prt_space}{self.fields["total_buffer_size [bytes]"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["total_buffer_size_on_mem [bytes]"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["avg_buffer_size_on_mem [bytes]"]}\n"
        result+= f"{prt_space}{prt_tee}{self.fields["total_buffer_size_on_compute [bytes]"]}\n"
        result+= f"{prt_space}{prt_last}{self.fields["avg_buffer_size_on_compute [bytes]"]}\n"
        result+= "\n"
        return result
    
    def _parse_runtime_results(self, log_file_path, regex):
        runtime = -1.0
        with open(log_file_path, 'r') as f:
            for line in f:
                match = re.search(regex, line)
                if match:
                    runtime = float(match.group(1))
                    break
        return runtime
    
    def _parse_build_stage_results(self):
        num_compute_tile = 0
        num_mem_tile = 0
        num_shim_tile = 0
        num_objectFIFO = 0
        num_unicast_objectFIFO = 0
        num_multicast_objectFIFO = 0
        num_objectFIFO_link = 0

        # Parse MLIR file for design info
        mlir_file_path = os.path.join(self.result_dir, "build", f"{self.benchmark_name}.mlir")
        if os.path.exists(mlir_file_path):
            with open(mlir_file_path, 'r') as file:
                for line in file:
                    # Count compute tiles
                    if re.search(r'\saie\.tile', line):
                        match = re.search(r'aie\.tile\((\d+),\s*(\d+)\)', line)
                        # Compute tile is at row_y >= 2
                        if int(match.group(2)) >= 2:
                            num_compute_tile += 1
                        # Memory tile is at row_y == 1
                        elif int(match.group(2)) == 1:
                            num_mem_tile += 1
                        # Shim tile is at row_y == 0
                        elif int(match.group(2)) == 0:
                            num_shim_tile += 1
                
                    # Count objectFIFO ops "aie.objectfifo"
                    if re.search(r'\saie\.objectfifo\s', line): 
                        num_objectFIFO += 1
                        # Check if this objectFIFO is unicast or multicast
                        match = re.search(r'\{([^\}]*)\}', line)
                        if match:
                            items = [item.strip() for item in match.group(1).split(',') if item.strip()]
                            if len(items) == 1:
                                num_unicast_objectFIFO += 1
                            elif len(items) > 1:
                                num_multicast_objectFIFO += 1
                    # Count objectFIFO link ops (e.g., "aie.objectfifo.link" or similar)
                    if re.search(r'\saie\.objectfifo\.link\s', line):
                        num_objectFIFO_link += 1
            # Update the fields
            self.fields["num_compute_tile"].value = num_compute_tile
            self.fields["num_mem_tile"].value = num_mem_tile
            self.fields["num_shim_tile"].value = num_shim_tile
            self.fields["num_objectFIFO"].value = num_objectFIFO
            self.fields["num_unicast_objectFIFO"].value = num_unicast_objectFIFO
            self.fields["num_multicast_objectFIFO"].value = num_multicast_objectFIFO
            self.fields["num_objectFIFO_link"].value = num_objectFIFO_link
        
        # Parse build log for build time
        build_log_file_path = os.path.join(self.result_dir, f"{self.task_name}.build.log")
        if os.path.exists(build_log_file_path):
            build_time = self._parse_runtime_results(build_log_file_path, r'Build took[:\s]*([0-9]+(?:\.[0-9]+)?)\s*(?:secs?|seconds?)')
            if build_time >= 0.0:
                self.fields["build_time [s]"].value = build_time

    def _parse_pnr_stage_results(self):
        # Parse PnR log
        pnr_log_file_path = os.path.join(self.result_dir, f"pnr.log")
        if os.path.exists(pnr_log_file_path):
            # Parse PnR runtime
            pnr_time = self._parse_runtime_results(pnr_log_file_path, r'Total placer runtime.*?:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:secs?|seconds?)')
            if pnr_time >= 0.0:
                self.fields["pnr_time [s]"].value = pnr_time
            
            num_SA_moves = 0
            num_SA_legal_moves = 0
            SA_legal_move_rate = 0.0
            num_SA_accepted_legal_moves = 0
            SA_legal_move_acceptance_rate = 0.0
            num_SA_rejected_legal_moves = 0
            SA_legal_move_rejection_rate = 0.0
            num_SA_congested_moves = 0
            SA_congested_move_rate = 0.0
            num_SA_accepted_congested_moves = 0
            SA_congested_move_acceptance_rate = 0.0
            num_SA_rejected_congested_moves = 0
            SA_congested_move_rejection_rate = 0.0
            num_SA_illegal_moves = 0
            SA_illegal_move_rate = 0.0
            num_SA_accepted_illegal_moves = 0
            SA_illegal_move_acceptance_rate = 0.0
            num_SA_rejected_illegal_moves = 0
            SA_illegal_move_rejection_rate = 0.0
            num_SA_accepted_moves = 0
            SA_move_acceptance_rate = 0.0
            num_SA_legal_accepted_moves = 0
            SA_accepted_move_legal_rate = 0.0
            num_SA_congested_accepted_moves = 0
            SA_accepted_move_congested_rate = 0.0
            num_SA_illegal_accepted_moves = 0
            SA_accepted_move_illegal_rate = 0.0
            num_SA_rejected_moves = 0
            SA_move_rejection_rate = 0.0
            num_SA_legal_rejected_moves = 0
            SA_rejected_move_legal_rate = 0.0
            num_SA_congested_rejected_moves = 0
            SA_rejected_move_congested_rate = 0.0
            num_SA_illegal_rejected_moves = 0
            SA_rejected_move_illegal_rate = 0.0
            current_best_cost = sys.float_info.max
            SA_per_move_info = []
            # Parse Simulated Annealing stats
            with open(pnr_log_file_path, 'r') as file:
                for line in file:
                    # Extract SA move statistics
                    # Total move attempts
                    match = re.search(r'\[SA\] Total # of move attempts:\s*([0-9]+)', line)
                    if match:
                        num_SA_moves = int(match.group(1))
                    
                    # Total legal/congested/illegal/accepted/rejected moves
                    match = re.search(r'Legal moves:\s*([0-9]+)', line)
                    if match:
                        num_SA_legal_moves = int(match.group(1))
                    
                    match = re.search(r'Congested moves:\s*([0-9]+)', line)
                    if match:
                        num_SA_congested_moves = int(match.group(1))
                    
                    match = re.search(r'Fatal moves:\s*([0-9]+)', line)
                    if match:
                        num_SA_illegal_moves = int(match.group(1))
                    
                    match = re.search(r'Accepted moves:\s*([0-9]+)', line)
                    if match:
                        num_SA_accepted_moves = int(match.group(1))
                    
                    match = re.search(r'Rejected moves:\s*([0-9]+)', line)
                    if match:
                        num_SA_rejected_moves = int(match.group(1))
                    
                    # Count accepted legal moves
                    match = re.search(r'\[SA\] ACCEPT Move:.* Legality: Legal', line)
                    if match:
                        num_SA_accepted_legal_moves += 1
                        num_SA_legal_accepted_moves += 1
                    
                    # Count accepted congested moves
                    match = re.search(r'\[SA\] ACCEPT Move:.* Legality: Congested', line)
                    if match:
                        num_SA_accepted_congested_moves += 1
                        num_SA_congested_accepted_moves += 1
                    
                    # Count accepted illegal moves
                    match = re.search(r'\[SA\] ACCEPT Move:.* Legality: Fatal', line)
                    if match:
                        num_SA_accepted_illegal_moves += 1
                        num_SA_illegal_accepted_moves += 1

                    # Count rejected legal moves
                    match = re.search(r'\[SA\] REJECT Move:.* Legality: Legal', line)
                    if match:
                        num_SA_rejected_legal_moves += 1
                        num_SA_legal_rejected_moves += 1

                    # Count rejected congested moves
                    match = re.search(r'\[SA\] REJECT Move:.* Legality: Congested', line)
                    if match:
                        num_SA_rejected_congested_moves += 1
                        num_SA_congested_rejected_moves += 1

                    # Count rejected illegal moves
                    match = re.search(r'\[SA\] REJECT Move:.* Legality: Fatal', line)
                    if match:
                        num_SA_rejected_illegal_moves += 1
                        num_SA_illegal_rejected_moves += 1
                    
                    # Find the current best cost
                    match = re.search(r'\[SA\] Initial cost:\s*([0-9.]+)', line)
                    if match:
                        current_best_cost = float(match.group(1))
                    match = re.search(r'\[SA\] .* best cost\s*([0-9.]+)', line)
                    if match:
                        current_best_cost = float(match.group(1))

                    # Store per-move info
                    match = re.search(r'(ACCEPT|REJECT).*?Cur Cost:\s*([0-9.]+).*?Legality:\s*(Legal|Congested|Fatal).*?T:\s*([0-9.]+)', line)
                    if match:
                        current_cost = float(match.group(2))
                        acceptance = match.group(1)
                        legality = match.group(3)
                        temperature = float(match.group(4))
                        delta_cost = current_cost - current_best_cost
                        accepted_legal_move_delta_cost = ""
                        accepted_congested_move_delta_cost = ""
                        accepted_illegal_move_delta_cost = ""
                        rejected_legal_move_delta_cost = ""
                        rejected_congested_move_delta_cost = ""
                        rejected_illegal_move_delta_cost = ""
                        delta_cost_90th_percentile = 0.0
                        delta_cost_75th_percentile = 0.0
                        delta_cost_50th_percentile = 0.0
                        delta_cost_25th_percentile = 0.0
                        delta_cost_10th_percentile = 0.0

                        if acceptance == "ACCEPT":
                            if legality == "Legal":
                                accepted_legal_move_delta_cost = delta_cost
                            elif legality == "Congested":
                                accepted_congested_move_delta_cost = delta_cost
                            elif legality == "Fatal":
                                accepted_illegal_move_delta_cost = delta_cost
                        elif acceptance == "REJECT":
                            if legality == "Legal":
                                rejected_legal_move_delta_cost = delta_cost
                            elif legality == "Congested":
                                rejected_congested_move_delta_cost = delta_cost
                            elif legality == "Fatal":
                                rejected_illegal_move_delta_cost = delta_cost
                        
                        if temperature > 0.0:
                            delta_cost_90th_percentile = math.log(0.9) * (-temperature)
                            delta_cost_75th_percentile = math.log(0.75) * (-temperature)
                            delta_cost_50th_percentile = math.log(0.5) * (-temperature)
                            delta_cost_25th_percentile = math.log(0.25) * (-temperature)
                            delta_cost_10th_percentile = math.log(0.1) * (-temperature)

                        SA_per_move_info.append({
                            "move_number": len(SA_per_move_info) + 1,
                            "best_cost": current_best_cost,
                            "move_cost": current_cost,
                            "acceptance": acceptance,
                            "legality": legality,
                            "temperature": temperature,
                            "delta_cost": delta_cost,
                            "accepted_legal_move_delta_cost": accepted_legal_move_delta_cost,
                            "accepted_congested_move_delta_cost": accepted_congested_move_delta_cost,
                            "accepted_illegal_move_delta_cost": accepted_illegal_move_delta_cost,
                            "rejected_legal_move_delta_cost": rejected_legal_move_delta_cost,
                            "rejected_congested_move_delta_cost": rejected_congested_move_delta_cost,
                            "rejected_illegal_move_delta_cost": rejected_illegal_move_delta_cost,
                            "delta_cost_10th_percentile": delta_cost_10th_percentile,
                            "delta_cost_25th_percentile": delta_cost_25th_percentile,
                            "delta_cost_50th_percentile": delta_cost_50th_percentile,
                            "delta_cost_75th_percentile": delta_cost_75th_percentile,
                            "delta_cost_90th_percentile": delta_cost_90th_percentile,
                        })

                # Calculate accepted/rejected move breakdowns
                if num_SA_moves > 0:
                    SA_legal_move_rate = (num_SA_legal_moves / num_SA_moves) * 100.0
                    SA_congested_move_rate = (num_SA_congested_moves / num_SA_moves) * 100.0
                    SA_illegal_move_rate = (num_SA_illegal_moves / num_SA_moves) * 100.0
                    SA_move_acceptance_rate = (num_SA_accepted_moves / num_SA_moves) * 100.0
                    SA_move_rejection_rate = (num_SA_rejected_moves / num_SA_moves) * 100.0

                if num_SA_legal_moves > 0:
                    SA_legal_move_acceptance_rate = (num_SA_accepted_legal_moves / num_SA_legal_moves) * 100.0
                    SA_legal_move_rejection_rate = (num_SA_rejected_legal_moves / num_SA_legal_moves) * 100.0
                if num_SA_congested_moves > 0:
                    SA_congested_move_acceptance_rate = (num_SA_accepted_congested_moves / num_SA_congested_moves) * 100.0
                    SA_congested_move_rejection_rate = (num_SA_rejected_congested_moves / num_SA_congested_moves) * 100.0
                if num_SA_illegal_moves > 0:
                    SA_illegal_move_acceptance_rate = (num_SA_accepted_illegal_moves / num_SA_illegal_moves) * 100.0
                    SA_illegal_move_rejection_rate = (num_SA_rejected_illegal_moves / num_SA_illegal_moves) * 100.0
                if num_SA_accepted_moves > 0:
                    SA_accepted_move_legal_rate = (num_SA_legal_accepted_moves / num_SA_accepted_moves) * 100.0
                    SA_accepted_move_congested_rate = (num_SA_congested_accepted_moves / num_SA_accepted_moves) * 100.0
                    SA_accepted_move_illegal_rate = (num_SA_illegal_accepted_moves / num_SA_accepted_moves) * 100.0
                if num_SA_rejected_moves > 0:
                    SA_rejected_move_legal_rate = (num_SA_legal_rejected_moves / num_SA_rejected_moves) * 100.0
                    SA_rejected_move_congested_rate = (num_SA_congested_rejected_moves / num_SA_rejected_moves) * 100.0
                    SA_rejected_move_illegal_rate = (num_SA_illegal_rejected_moves / num_SA_rejected_moves) * 100.0

            # Update the fields
            self.fields["num_SA_moves"].value = num_SA_moves
            self.fields["num_SA_legal_moves"].value = num_SA_legal_moves
            self.fields["SA_legal_move_rate [%]"].value = SA_legal_move_rate
            self.fields["num_SA_congested_moves"].value = num_SA_congested_moves
            self.fields["SA_congested_move_rate [%]"].value = SA_congested_move_rate
            self.fields["num_SA_illegal_moves"].value = num_SA_illegal_moves
            self.fields["SA_illegal_move_rate [%]"].value = SA_illegal_move_rate
            self.fields["num_SA_accepted_moves"].value = num_SA_accepted_moves
            self.fields["SA_move_acceptance_rate [%]"].value = SA_move_acceptance_rate
            self.fields["num_SA_rejected_moves"].value = num_SA_rejected_moves
            self.fields["SA_move_rejection_rate [%]"].value = SA_move_rejection_rate

            self.fields["num_SA_accepted_legal_moves"].value = num_SA_accepted_legal_moves
            self.fields["SA_legal_move_acceptance_rate [%]"].value = SA_legal_move_acceptance_rate
            self.fields["num_SA_rejected_legal_moves"].value = num_SA_rejected_legal_moves
            self.fields["SA_legal_move_rejection_rate [%]"].value = SA_legal_move_rejection_rate
            self.fields["num_SA_accepted_congested_moves"].value = num_SA_accepted_congested_moves
            self.fields["SA_congested_move_acceptance_rate [%]"].value = SA_congested_move_acceptance_rate
            self.fields["num_SA_rejected_congested_moves"].value = num_SA_rejected_congested_moves
            self.fields["SA_congested_move_rejection_rate [%]"].value = SA_congested_move_rejection_rate
            self.fields["num_SA_accepted_illegal_moves"].value = num_SA_accepted_illegal_moves
            self.fields["SA_illegal_move_acceptance_rate [%]"].value = SA_illegal_move_acceptance_rate
            self.fields["num_SA_rejected_illegal_moves"].value = num_SA_rejected_illegal_moves
            self.fields["SA_illegal_move_rejection_rate [%]"].value = SA_illegal_move_rejection_rate

            self.fields["num_SA_legal_accepted_moves"].value = num_SA_legal_accepted_moves
            self.fields["SA_accepted_move_legal_rate [%]"].value = SA_accepted_move_legal_rate
            self.fields["num_SA_congested_accepted_moves"].value = num_SA_congested_accepted_moves
            self.fields["SA_accepted_move_congested_rate [%]"].value = SA_accepted_move_congested_rate
            self.fields["num_SA_illegal_accepted_moves"].value = num_SA_illegal_accepted_moves
            self.fields["SA_accepted_move_illegal_rate [%]"].value = SA_accepted_move_illegal_rate
            self.fields["num_SA_legal_rejected_moves"].value = num_SA_legal_rejected_moves
            self.fields["SA_rejected_move_legal_rate [%]"].value = SA_rejected_move_legal_rate
            self.fields["num_SA_congested_rejected_moves"].value = num_SA_congested_rejected_moves
            self.fields["SA_rejected_move_congested_rate [%]"].value = SA_rejected_move_congested_rate
            self.fields["num_SA_illegal_rejected_moves"].value = num_SA_illegal_rejected_moves
            self.fields["SA_rejected_move_illegal_rate [%]"].value = SA_rejected_move_illegal_rate

            self.fields["SA_per_move_info"].value = SA_per_move_info

    def _parse_aiecc_compilation_stage_results(self):
        # Parse AIECC compilation log for compilation time
        aiecc_compile_log_file_path = os.path.join(self.result_dir, f"{self.task_name}.compile.log")
        if os.path.exists(aiecc_compile_log_file_path):
            compilation_time = self._parse_runtime_results(aiecc_compile_log_file_path, r'Compile took[:\s]*([0-9]+(?:\.[0-9]+)?)\s*(?:secs?|seconds?)')
            if compilation_time >= 0.0:
                self.fields["compilation_time [s]"].value = compilation_time
        
        # Calculate total end-to-end compilation time
        total_time = 0.0
        build_time = self.fields["build_time [s]"].value
        compilation_time = self.fields["compilation_time [s]"].value
        if build_time != "N/A" and float(build_time) >= 0.0:
            total_time += float(build_time)
        if compilation_time != "N/A" and float(compilation_time) >= 0.0:
            total_time += float(compilation_time)
        if total_time > 0.0:
            self.fields["total_end2end_compilation_time [s]"].value = total_time
        
        # Parse the final placement of the design from routing summary JSON file
        routing_summary_json_file_path = os.path.join(self.result_dir, "build", f"post_compile_routing_summary.json")
        if os.path.exists(routing_summary_json_file_path):
            with open(routing_summary_json_file_path, 'r') as file:
                data = json.load(file)

            # Count neighbor connections
            nbr_route_count = 0
            if 'nbr_routes' in data:
                nbr_route_count = len(data['nbr_routes'])

            # Count circuit switch connections
            cct_route_count = 0
            if 'cct_routes' in data:
                cct_route_count = len(data['cct_routes'])

            # Count packet flow connections
            pkt_route_count = 0
            if 'pkt_routes' in data:
                pkt_route_count = len(data['pkt_routes'])
            
            # Count number of circuit switch track used
            total_cct_route_tracks = 0
            largest_cct_route_net = 0
            longest_cct_route_path = 0
            smallest_cct_route_net = sys.maxsize
            shortest_cct_route_path = sys.maxsize
            num_tracks_per_cct_route_net = []
            avg_num_tracks_per_cct_route_net = 0
            for cct_route in data.get('cct_routes', []):
                unique_tracks = set()   # account for track sharing in multicast
                for path in cct_route.get("intermediates", []):
                    for node_idx, node in enumerate(path[:-1]):
                        segment_id = (node["col_x"], node["row_y"], path[node_idx + 1]["col_x"], path[node_idx + 1]["row_y"])
                        unique_tracks.add(segment_id)
                total_cct_route_tracks += len(unique_tracks)
                largest_cct_route_net = max(largest_cct_route_net, len(unique_tracks))
                longest_cct_route_path = max(longest_cct_route_path, len(path)-1)
                smallest_cct_route_net = min(smallest_cct_route_net, len(unique_tracks))
                shortest_cct_route_path = min(shortest_cct_route_path, len(path)-1)
                num_tracks_per_cct_route_net.append(len(unique_tracks))

            if num_tracks_per_cct_route_net:
                avg_num_tracks_per_cct_route_net = sum(num_tracks_per_cct_route_net) / len(num_tracks_per_cct_route_net)

            # Count number of packet flow track used
            total_pkt_route_tracks = 0
            largest_pkt_route_net = 0
            longest_pkt_route_path = 0
            smallest_pkt_route_net = sys.maxsize
            shortest_pkt_route_path = sys.maxsize
            num_tracks_per_pkt_route_net = []
            avg_num_tracks_per_pkt_route_net = 0
            for pkt_route in data.get('pkt_routes', []):
                unique_tracks = set()   # account for track sharing in multicast
                for path in pkt_route.get("intermediates", []):
                    for node_idx, node in enumerate(path[:-1]):
                        segment_id = (node["col_x"], node["row_y"], path[node_idx + 1]["col_x"], path[node_idx + 1]["row_y"])
                        unique_tracks.add(segment_id)
                total_pkt_route_tracks += len(unique_tracks)
                largest_pkt_route_net = max(largest_pkt_route_net, len(unique_tracks))
                longest_pkt_route_path = max(longest_pkt_route_path, len(path)-1)
                smallest_pkt_route_net = min(smallest_pkt_route_net, len(unique_tracks))
                shortest_pkt_route_path = min(shortest_pkt_route_path, len(path)-1)
                num_tracks_per_pkt_route_net.append(len(unique_tracks))

            if num_tracks_per_pkt_route_net:
                avg_num_tracks_per_pkt_route_net = sum(num_tracks_per_pkt_route_net) / len(num_tracks_per_pkt_route_net)

            # Count number of DMA port Usage
            total_dma_ports = 0
            total_dma_in_ports = 0
            total_dma_out_ports = 0
            dma_ports = dict()
            for cct_route in data.get('cct_routes', []):
                x = cct_route["src"]["col_x"]
                y = cct_route["src"]["row_y"]
                total_dma_ports += 1
                total_dma_out_ports += 1
                if (x, y) in dma_ports:
                    dma_ports[(x, y)]["out_count"] += 1
                else:
                    dma_ports[(x, y)] = dict(in_count=0, out_count=1)

                for dst in cct_route["dsts"]:
                    x = dst["col_x"]
                    y = dst["row_y"]
                    total_dma_ports += 1
                    total_dma_in_ports += 1
                    if (x, y) in dma_ports:
                        dma_ports[(x, y)]["in_count"] += 1
                    else:
                        dma_ports[(x, y)] = dict(in_count=1, out_count=0)
                    
            # Calculate total buffer size
            buffer_on_mem = []
            buffer_on_compute = []
            total_buffer_size = 0
            total_buffer_on_mem = 0
            total_buffer_on_compute = 0
            avg_buffer_size_on_mem = 0
            avg_buffer_size_on_compute = 0
            for buffer in data['buffers']:
                total_buffer_size += buffer.get('total_size_bytes', 0)
                if buffer.get('row_y') == 1:
                    buffer_on_mem.append(buffer.get('total_size_bytes', 0))
                elif buffer.get('row_y') > 1:
                    buffer_on_compute.append(buffer.get('total_size_bytes', 0))
            total_buffer_on_mem = sum(buffer_on_mem)
            total_buffer_on_compute = sum(buffer_on_compute)
            if buffer_on_mem:
                avg_buffer_size_on_mem = total_buffer_on_mem / len(buffer_on_mem)
            if buffer_on_compute:
                avg_buffer_size_on_compute = total_buffer_on_compute / len(buffer_on_compute)

            # Update the fields
            self.fields["num_neighbour_sharing_objectFIFO"].value = nbr_route_count
            self.fields["num_circuit_switch_objectFIFO"].value = cct_route_count
            self.fields["num_packet_flow_objectFIFO"].value = pkt_route_count
            if cct_route_count > 0:
                self.fields["total_num_circuit_switch_tracks"].value = total_cct_route_tracks
                self.fields["largest_circuit_switch_net"].value = largest_cct_route_net
                self.fields["smallest_circuit_switch_net"].value = smallest_cct_route_net
                self.fields["avg_track_per_circuit_switch_net"].value = avg_num_tracks_per_cct_route_net
                self.fields["longest_circuit_switch_path"].value = longest_cct_route_path
                self.fields["shortest_circuit_switch_path"].value = shortest_cct_route_path
            if pkt_route_count > 0:
                self.fields["total_num_packet_flow_tracks"].value = total_pkt_route_tracks
                self.fields["largest_packet_flow_net"].value = largest_pkt_route_net
                self.fields["smallest_packet_flow_net"].value = smallest_pkt_route_net
                self.fields["avg_track_per_packet_flow_net"].value = avg_num_tracks_per_pkt_route_net
                self.fields["longest_packet_flow_path"].value = longest_pkt_route_path
                self.fields["shortest_packet_flow_path"].value = shortest_pkt_route_path
            self.fields["total_num_dma_ports"].value = total_dma_ports
            self.fields["total_num_dma_in_ports"].value = total_dma_in_ports
            self.fields["total_num_dma_out_ports"].value = total_dma_out_ports
            self.fields["total_buffer_size [bytes]"].value = total_buffer_size
            self.fields["total_buffer_size_on_mem [bytes]"].value = total_buffer_on_mem
            self.fields["avg_buffer_size_on_mem [bytes]"].value = avg_buffer_size_on_mem
            self.fields["total_buffer_size_on_compute [bytes]"].value = total_buffer_on_compute
            self.fields["avg_buffer_size_on_compute [bytes]"].value = avg_buffer_size_on_compute

    def _parse_npu_run_stage_results(self):
        # Parse NPU run log for average runtime
        npu_run_log_file_path = os.path.join(self.result_dir, f"{self.task_name}.run.log")
        if os.path.exists(npu_run_log_file_path):
            avg_runtime = self._parse_runtime_results(npu_run_log_file_path, r'Avg NPU time:\s*([\d.]+)\s*us')
            if avg_runtime >= 0.0:
                self.fields["avg_NPU_runtime [us]"].value = avg_runtime

    def get_field_names(self):
        field_names = []
        for field in self.fields.values():
            if not field.hidden_field:
                field_names.append(field.name)
        return field_names
    
    def get_field_values(self):
        field_values = []
        for field in self.fields.values():
            if not field.hidden_field:
                field_values.append(str(field.value))
        return field_values
    
    def get_field_names_as_csv(self):
        return ",".join(self.get_field_names())

    def get_field_values_as_csv(self):
        return ",".join(self.get_field_values())
    
    def get_per_SA_move_info(self):
        return self.SA_per_SA_move_info
    
    def get_per_SA_move_info_as_csv(self):
        csv_lines = []
        if self.fields["SA_per_move_info"].value:
            header = ",".join(self.fields["SA_per_move_info"].value[0].keys())
            csv_lines.append(header)
            for move_info in self.fields["SA_per_move_info"].value:
                line = ",".join(str(value) for value in move_info.values())
                csv_lines.append(line)
            return "\n".join(csv_lines)
        else:
            return ""
    
    def generate_SA_move_info_excel_with_plot(self):
        # Helper function to create scatter plot series
        def create_scatter_plot_series(ws, yvalue_col, xvalue, marker_shape, maker_size=5, color="000000", line_no_fill=True):
            yvalue = Reference(ws, min_col=yvalue_col, min_row=1, max_row=ws.max_row)
            series = Series(yvalue, xvalue, title_from_data=True)
            series.graphicalProperties.line.noFill = line_no_fill
            if line_no_fill:
                series.marker.symbol = marker_shape
                series.marker.size = maker_size
                series.marker.graphicalProperties.noFill = True
                # series.marker.graphicalProperties.solidFill = color
                series.marker.graphicalProperties.line.solidFill = color
            else:
                series.graphicalProperties.line.solidFill = color
            return series
    
        # Create a workbook and write values using openpyxl
        wb = Workbook()
        ws = wb.active
        ws.title = "SA Move Info"

        # Read CSV into Excel
        SA_move_info = self.get_per_SA_move_info_as_csv()
        reader = csv.reader(SA_move_info.splitlines())
        for row in reader:
            # Convert numeric strings with commas to floats
            new_row = []
            for cell in row:
                try:
                    new_row.append(float(cell))
                except Exception:
                    new_row.append(cell)
            ws.append(new_row)
        # If no data, return
        if ws.max_row < 2:
            return

        # Create a Scatter chart for absolute cost of each SA move
        marker_size = 6
        font_size = 16
        font_type = Font(typeface='Calibri')

        chart = ScatterChart()
        chart.title = "Cost of each SA move"
        chart.style = 13
        chart.x_axis.title = "Number of SA moves"
        chart.x_axis.tickLblPos = "nextTo"
        chart.x_axis.delete = False
        chart.y_axis.title = "Cost"
        chart.x_axis.tickLblPos = "nextTo"
        chart.y_axis.delete = False
        chart.legend.position = 'r'
        chart.height = 20
        chart.width = 30
        chart.layout=Layout(manualLayout=ManualLayout(
            x=0, y=0, h=0.8, w=0.8,
        ))

        cp = CharacterProperties(latin=font_type, sz=font_size * 100)
        pp = ParagraphProperties(defRPr=cp)
        rtp = RichText(p=[Paragraph(pPr=pp, endParaRPr=cp)])
        chart.title.tx.rich.p[0].pPr = pp
        chart.x_axis.txPr = rtp
        chart.x_axis.title.tx.rich.p[0].pPr = pp
        chart.y_axis.txPr = rtp
        chart.y_axis.title.tx.rich.p[0].pPr = pp
        chart.legend.txPr = rtp

        xvalues = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
        # Column B (index 2) is current best cost at each move
        chart.series.append(create_scatter_plot_series(ws, 2, xvalues, "circle", marker_size, "0000FF", line_no_fill=False))
        # Column C (index 3) is current cost at each move
        chart.series.append(create_scatter_plot_series(ws, 3, xvalues, "triangle", marker_size, "00FF00"))
        ws.add_chart(chart, "T2")  # Place chart at cell T2

        # Create a Scatter chart for delta cost of each SA move
        chart = ScatterChart()
        chart.title = "Detla cost of each SA move"
        chart.style = 13
        chart.x_axis.title = "Number of SA moves"
        chart.x_axis.tickLblPos = "nextTo"
        chart.x_axis.delete = False
        chart.y_axis.title = "Delta Cost"
        chart.x_axis.tickLblPos = "nextTo"
        chart.y_axis.delete = False
        chart.legend.position = 'b'
        chart.height = 20
        chart.width = 40
        chart.layout=Layout(manualLayout=ManualLayout(
            x=-0.125, y=0, h=0.8, w=0.60,
        ))
        chart.legend.layout=Layout(manualLayout=ManualLayout(
            xMode='edge',
            yMode='edge',
            x=0.7, y=0.1, h=0.8, w=0.3,
        ))

        cp = CharacterProperties(latin=font_type, sz=font_size * 100)
        pp = ParagraphProperties(defRPr=cp)
        rtp = RichText(p=[Paragraph(pPr=pp, endParaRPr=cp)])
        chart.title.tx.rich.p[0].pPr = pp
        chart.x_axis.txPr = rtp
        chart.x_axis.title.tx.rich.p[0].pPr = pp
        chart.y_axis.txPr = rtp
        chart.y_axis.title.tx.rich.p[0].pPr = pp
        chart.legend.txPr = rtp

        xvalues = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
        # Column H (index 8) is accepted legal moves delta cost
        chart.series.append(create_scatter_plot_series(ws, 8, xvalues, "circle", marker_size, "00FF00"))
        # Column I (index 9) is accepted congested moves delta cost
        chart.series.append(create_scatter_plot_series(ws, 9, xvalues, "triangle", marker_size, "00FF00"))
        # Column J (index 10) is accepted illegal moves delta cost
        chart.series.append(create_scatter_plot_series(ws, 10, xvalues, "x", marker_size, "00FF00"))
        # Column K (index 11) is rejected legal moves delta cost
        chart.series.append(create_scatter_plot_series(ws, 11, xvalues, "circle", marker_size, "FF0000"))
        # Column L (index 12) is rejected congested moves delta cost
        chart.series.append(create_scatter_plot_series(ws, 12, xvalues, "triangle", marker_size, "FF0000"))
        # Column M (index 13) is rejected illegal moves delta cost
        chart.series.append(create_scatter_plot_series(ws, 13, xvalues, "x", marker_size, "FF0000"))

        # Column N (index 14) is 10th percentile delta cost line
        chart.series.append(create_scatter_plot_series(ws, 14, xvalues, "dash", marker_size, "FF7000", line_no_fill=False))
        # Column O (index 15) is 25th percentile delta cost line
        chart.series.append(create_scatter_plot_series(ws, 15, xvalues, "dash", marker_size, "FFAD00", line_no_fill=False))
        # Column P (index 16) is 50th percentile delta cost line
        chart.series.append(create_scatter_plot_series(ws, 16, xvalues, "dash", marker_size, "FFFF00", line_no_fill=False))
        # Column Q (index 17) is 75th percentile delta cost line
        chart.series.append(create_scatter_plot_series(ws, 17, xvalues, "dash", marker_size, "ADFF00", line_no_fill=False))
        # Column R (index 18) is 90th percentile delta cost line
        chart.series.append(create_scatter_plot_series(ws, 18, xvalues, "dash", marker_size, "70FF00", line_no_fill=False))

        ws.add_chart(chart, "T42")  # Place chart at cell T48

        # Create a separate Scatter chart for delta cost of each SA move in log scale
        chart = copy.deepcopy(chart)
        chart.title = "Detla cost of each SA move (Log Scale)"
        chart.y_axis.title = "Delta Cost (Log Scale)"
        chart.y_axis.scaling.logBase = 10
        ws.add_chart(chart, "T82")  # Place chart at cell T48

        # Save workbook
        wb.save(os.path.join(self.result_dir, f"{self.task_name}_SA_per_move_info.xlsx"))


# This class represents the collection of test results
class ResultTable:
    def __init__(self, tasklist: list, result_dir: str, output_csv: str):
        self.tasklist_entries = self._parse_tasklist_entry(tasklist)
        self.result_dir = result_dir
        self.output_csv = output_csv
        self.results = self._load_results(result_dir)
        self.headers = self._load_headers()
    
    def __repr__(self):
        result = "ResultTable:\n"
        for test_result in self.results.values():
            result += str(test_result)
        return result
    
    # Parse the tasklist entries which can be file paths or direct benchmark/task entries
    def _parse_tasklist_entry(self, tasklist: list):
        task_entries = []
        for arg_item in tasklist:
            # Check if the argument is a tasklist file
            # If it is a file, read the benchmark/task entries from the file
            if os.path.isfile(arg_item):
                with open(arg_item, "r") as tasklist_file:
                    for task in yaml.safe_load(tasklist_file):
                        task_entries.append(task)
            # If it is not a file, treat it as a benchmark/task entry
            else:
                task_entries.append(arg_item)
        return task_entries

    # Load the results from the result directory
    def _load_results(self, result_dir: str):
        results = OrderedDict()
        # For each benchmark/task entry, create a TaskResult object
        for entry in self.tasklist_entries:
            benchmark_task_pair = entry.split("/")
            assert len(benchmark_task_pair) == 2, f"Invalid task entry '{entry}'. Expected format: benchmark/task"

            benchmark_name, task_name = benchmark_task_pair
            task_result_dir = os.path.join(result_dir, benchmark_name, task_name)
            results[f"{benchmark_name}/{task_name}"] = TestResult(task_result_dir, benchmark_name, task_name)
        return results

    # Load the headers from the first result
    def _load_headers(self):
        if self.results:
            first_result = next(iter(self.results.values()))
            return first_result.get_field_names()
        else:
            return []

    # Write the results to the output CSV file
    def write_to_csv(self):
        with open(self.output_csv, "w") as csv_file:
            # Write the header
            csv_file.write(",".join(self.headers) + "\n")
            for test_result in self.results.values():
                csv_file.write(test_result.get_field_values_as_csv() + "\n")

    def organize_SA_per_move_info_to_excel(self):
        for test_result in self.results.values():
            test_result.generate_SA_move_info_excel_with_plot()


def collect_results(tasklist, result_dir, output_csv = 'results.csv', verbose=False):
    results = ResultTable(tasklist, result_dir, output_csv)
    results.write_to_csv()
    results.organize_SA_per_move_info_to_excel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parse benchmark results and generate a summary CSV file.")
    parser.add_argument(
        "--tasklists",
        type=str,
        nargs="*",
        required=True,
        help="Path to the task list file or benchmark/task pairs (e.g. vector_scalar_add/default)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing benchmark output folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results.csv",
        required=False,
        help="Output CSV file path.",
    )
    args = parser.parse_args()
    
    collect_results(args.tasklists, args.input_dir, args.output_csv)