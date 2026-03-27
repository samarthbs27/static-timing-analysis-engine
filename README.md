# Static Timing Analysis (STA) Tool

A command-line Static Timing Analysis engine written in pure Python (stdlib only, no graph libraries used).
It parses a gate-level netlist (`.bench`) and an NLDM Liberty library (`.lib`),
then computes arrival times, slews, required arrival times, slack values, and the
critical path using bilinear LUT interpolation.

## Features

- **Netlist parsing** — reads `.bench` format; builds fanin/fanout graph
- **Liberty parsing** — extracts NLDM `cell_delay` and `output_slew` 2-D LUTs from `.lib` files
- **Load capacitance computation** — sums input caps of gate fanouts; uses 4× inverter cap for primary outputs
- **Forward traversal** — propagates arrival times and output slews gate-by-gate in topological order
- **Backward traversal** — propagates required arrival times; computes slack at every gate and primary I/O
- **Critical path tracing** — follows minimum-slack fanin back from the worst primary output to a primary input
- **No external dependencies** — uses only Python standard library (`dataclasses`, `re`, `pathlib`, `argparse`)

## Requirements

- Python 3.7+

## Usage

### Circuit details

Parse a netlist and print gate counts, fanin, and fanout lists.

```bash
python main_sta.py --read_ckt <circuit.bench>
```

Output written to `ckt_details.txt`.

### Library LUT inspection

Print the delay or slew look-up tables from a Liberty file.

```bash
python main_sta.py --read_nldm <library.lib> --delays
python main_sta.py --read_nldm <library.lib> --slews
```

Output written to `delay_LUT.txt` or `slew_LUT.txt` respectively.

### Full Static Timing Analysis

Run STA with both a netlist and a Liberty library.

```bash
python main_sta.py --read_ckt <circuit.bench> --read_nldm <library.lib>
```

Output written to `ckt_traversal.txt` containing:
- Circuit delay (ps)
- Slack at every primary input, primary output, and internal gate (ps)
- Critical path (comma-separated list of node labels from PI to PO)

## Example

```bash
# Inspect the ISCAS-85 c17 benchmark
python main_sta.py --read_ckt c17.bench

# View delay LUTs from the sample library
python main_sta.py --read_nldm sample_NLDM.lib --delays

# Run full STA on c17
python main_sta.py --read_ckt c17.bench --read_nldm sample_NLDM.lib
```

## Input File Formats

### `.bench` (gate-level netlist)

```
INPUT(A)
INPUT(B)
OUTPUT(Z)
n1 = NAND(A, B)
Z  = NOT(n1)
```

Supported gate keywords: `NAND`, `NOR`, `AND`, `OR`, `NOT`/`INV`, `BUF`, and any other
type present in the Liberty library.

### `.lib` (NLDM Liberty subset)

Standard Synopsys Liberty format. The parser extracts:
- `cell_delay` timing table
- `output_slew` timing table
- `capacitance` from input pin blocks

## Output Files

| File | Contents |
|------|----------|
| `ckt_details.txt` | Primary I/O counts, gate type counts, fanout list, fanin list |
| `delay_LUT.txt` | Cell delay 2-D LUT per cell (input slew × load cap) |
| `slew_LUT.txt` | Output slew 2-D LUT per cell |
| `ckt_traversal.txt` | Circuit delay, all gate/PI/PO slacks, critical path |

## Included Sample Files

| File | Description |
|------|-------------|
| `c17.bench` | ISCAS-85 c17 benchmark (5 inputs, 2 outputs, 6 NAND gates) |
| `b15.bench` | ITC-99 b15 benchmark |
| `c7552.bench` | ISCAS-85 c7552 benchmark |
| `sample_NLDM.lib` | NLDM Liberty library with delay/slew LUTs |

## Algorithm Overview

1. **Topological sort** — Kahn's algorithm on the fanin graph (no library used)
2. **Cload assignment** — per gate: Σ(input caps of gate fanouts) + 4 × INV_cap × (# PO fanouts)
3. **Forward pass** — for each gate in topo order:
   - Look up `cell_delay` and `output_slew` via bilinear interpolation
   - Scale by `n/2` for gates with more than 2 inputs
   - `max_out_arrival = max(inp_arrival[i] + cell_delay[i])`
4. **Backward pass** — required arrival time = 1.1 × circuit delay at every PO;
   propagated backward as `req_arrival[gate] = min(fanout.req_arrival − fanout.cell_delay[i])`
5. **Slack** — `(req_arrival − max_out_arrival) × 1000` (converted to ps)
6. **Critical path** — greedy backward trace from the minimum-slack PO, following the minimum-slack fanin at each step
