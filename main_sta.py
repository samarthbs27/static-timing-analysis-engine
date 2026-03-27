#!/usr/bin/env python3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union, Optional
import argparse
import re


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Node:
    out_net: str
    gate_type: str
    in_nets: List[str]

    fanin:  List[Union["Node", str]] = field(default_factory=list)   # Node or "INPUT-x"
    fanout: List[Union["Node", str]] = field(default_factory=list)   # Node or "OUTPUT-y"

    # STA fields (populated during Phase-2)
    Cload:        float       = 0.0
    Tau_in:       List[float] = field(default_factory=list)   # input slews (ns)
    inp_arrival:  List[float] = field(default_factory=list)   # arrival at each input (ns)
    cell_delays:  List[float] = field(default_factory=list)   # delay per input path (ns)
    cell_slews:   List[float] = field(default_factory=list)   # output slew per input path (ns)
    outp_arrival: List[float] = field(default_factory=list)   # inp_arrival + cell_delay (ns)
    max_out_arrival: float    = 0.0                           # ns
    Tau_out:      float       = 0.0                           # output slew (ns)
    req_arrival:  float       = 0.0                           # required arrival at output (ns)
    slack:        float       = 0.0                           # ps

    def label(self) -> str:
        return f"{self.gate_type}-{self.out_net}"


@dataclass
class LUT:
    index_1_str:     str          # "0.001,0.004,..."
    index_2_str:     str          # "0.365,1.855,..."
    values_str_rows: List[str]    # each row as CSV string

    index_1: List[float]          # numeric tau values
    index_2: List[float]          # numeric C values
    values:  List[List[float]]    # 2-D table [tau_idx][C_idx]


@dataclass
class NLDMCell:
    name:        str
    cell_delay:  Optional[LUT] = None
    output_slew: Optional[LUT] = None
    input_cap:   float         = 0.0   # fF, from first input pin


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    import sys

    # Normalize Unicode dashes to ASCII hyphens.
    # PDFs and Word documents often substitute en-dash (U+2013) or em-dash
    # (U+2014) for the double-hyphen "--" used in CLI flags.  This causes
    # argparse to reject arguments like "–-read_ckt".  We fix sys.argv in
    # place before argparse ever sees it.
    def _fix_arg(arg: str) -> str:
        for bad in ("\u2014", "\u2013", "\u2012", "\u2011", "\u2010"):
            if arg.startswith(bad):
                return "-" + arg[len(bad):]   # replace leading Unicode dash with one "-"
        return arg

    sys.argv = [sys.argv[0]] + [_fix_arg(a) for a in sys.argv[1:]]

    p = argparse.ArgumentParser()
    p.add_argument("--read_ckt",  type=Path, default=None, help="Path to .bench file")
    p.add_argument("--read_nldm", type=Path, default=None, help="Path to .lib file")
    p.add_argument("--delays",    action="store_true",     help="Print delay LUTs")
    p.add_argument("--slews",     action="store_true",     help="Print slew LUTs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Phase 2: both --read_ckt and --read_nldm supplied  →  STA traversal
    if args.read_ckt is not None and args.read_nldm is not None:
        pi, po, nodes, gate_order = read_ckt(args.read_ckt)
        nldm = read_nldm(args.read_nldm)
        run_sta(pi, po, nodes, gate_order, nldm, Path("ckt_traversal.txt"))
        return

    # Phase 1 Part A: only --read_ckt
    if args.read_ckt is not None:
        pi, po, nodes, gate_order = read_ckt(args.read_ckt)
        write_ckt_details(pi, po, nodes, gate_order, Path("ckt_details.txt"))
        return

    # Phase 1 Part B: only --read_nldm  (+  --delays or --slews)
    if args.read_nldm is not None:
        if args.delays == args.slews:          # both True or both False
            raise SystemExit("ERROR: Use exactly one of --delays or --slews with --read_nldm")
        nldm = read_nldm(args.read_nldm)
        if args.delays:
            write_delay_lut(nldm, Path("delay_LUT.txt"))
        else:
            write_slew_lut(nldm, Path("slew_LUT.txt"))
        return

    raise SystemExit("ERROR: Provide --read_ckt or --read_nldm")


# ---------------------------------------------------------------------------
# Phase-1 Part A: .bench parsing
# ---------------------------------------------------------------------------

def read_ckt(bench_path: Path):
    primary_inputs:  List[str] = []
    primary_outputs: List[str] = []
    nodes:      Dict[str, Node] = {}
    gate_order: List[str]       = []

    gate_re = re.compile(r"^\s*([^\s=]+)\s*=\s*([A-Za-z0-9_]+)\s*\(([^)]*)\)\s*$")

    with bench_path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("**"):
                continue

            if line.startswith("INPUT(") and line.endswith(")"):
                primary_inputs.append(line[len("INPUT("):-1].strip())
                continue

            if line.startswith("OUTPUT(") and line.endswith(")"):
                primary_outputs.append(line[len("OUTPUT("):-1].strip())
                continue

            m = gate_re.match(line)
            if not m:
                raise ValueError(f"Unrecognized line: {line}")

            out_net   = m.group(1).strip()
            gate_type = m.group(2).strip()
            in_blob   = m.group(3).strip()
            in_nets   = [t.strip() for t in in_blob.split(",") if t.strip()] if in_blob else []

            node = Node(out_net=out_net, gate_type=gate_type, in_nets=in_nets)
            nodes[out_net] = node
            gate_order.append(out_net)

    # Pass 2: wire fanin / fanout
    pi_set = set(primary_inputs)
    for out_net in gate_order:
        node = nodes[out_net]
        for in_net in node.in_nets:
            if in_net in pi_set:
                node.fanin.append(f"INPUT-{in_net}")
            else:
                if in_net not in nodes:
                    raise ValueError(f"Net '{in_net}' has no driver")
                driver = nodes[in_net]
                node.fanin.append(driver)
                driver.fanout.append(node)

    for po_net in primary_outputs:
        if po_net in nodes:
            nodes[po_net].fanout.append(f"OUTPUT-{po_net}")
        elif po_net not in pi_set:
            raise ValueError(f"Primary output '{po_net}' has no driver")

    return primary_inputs, primary_outputs, nodes, gate_order


def _ref_label(x: Union[Node, str]) -> str:
    return x if isinstance(x, str) else x.label()


def write_ckt_details(primary_inputs, primary_outputs, nodes, gate_order, out_path: Path):
    type_counts: Dict[str, int] = {}
    for out_net in gate_order:
        gt = nodes[out_net].gate_type
        type_counts[gt] = type_counts.get(gt, 0) + 1

    with out_path.open("w") as w:
        w.write(f"{len(primary_inputs)} primary inputs\n")
        w.write(f"{len(primary_outputs)} primary outputs\n")

        seen: set = set()
        for out_net in gate_order:
            gt = nodes[out_net].gate_type
            if gt in seen:
                continue
            seen.add(gt)
            w.write(f"{type_counts[gt]} {gt} gates\n")

        w.write("\nFanout...\n")
        for out_net in gate_order:
            node = nodes[out_net]
            w.write(f"{node.label()}: {', '.join(_ref_label(x) for x in node.fanout)}\n")

        w.write("\nFanin...\n")
        for out_net in gate_order:
            node = nodes[out_net]
            labels = [_ref_label(x) for x in node.fanin]
            pi_first = [s for s in labels if s.startswith("INPUT-")]
            rest     = [s for s in labels if not s.startswith("INPUT-")]
            w.write(f"{node.label()}: {', '.join(pi_first + rest)}\n")


# ---------------------------------------------------------------------------
# Phase-1 Part B: Liberty (.lib) parsing
# ---------------------------------------------------------------------------

def _find_matching_brace(s: str, open_pos: int) -> int:
    depth, in_str, esc = 0, False, False
    for i in range(open_pos, len(s)):
        ch = s[i]
        if in_str:
            if esc:   esc = False
            elif ch == "\\": esc = True
            elif ch == '"':  in_str = False
        else:
            if   ch == '"': in_str = True
            elif ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return i
    raise ValueError("Unmatched '{' in .lib file")


def _normalize_csv_string(s: str) -> str:
    toks = [t.strip() for t in s.strip().split(",") if t.strip()]
    return ",".join(toks)


def _csv_to_floats(csv_str: str) -> List[float]:
    return [float(t) for t in _normalize_csv_string(csv_str).split(",") if t]


def read_nldm(lib_path: Path) -> Dict[str, NLDMCell]:
    text = lib_path.read_text(errors="ignore")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*?$",    "", text, flags=re.MULTILINE)

    def parse_lut(table_block: str) -> LUT:
        m1 = re.search(r'index_1\s*\(\s*"([^"]+)"\s*\)\s*;', table_block, flags=re.DOTALL)
        m2 = re.search(r'index_2\s*\(\s*"([^"]+)"\s*\)\s*;', table_block, flags=re.DOTALL)
        mv = re.search(r"values\s*\(\s*(.*?)\s*\)\s*;",       table_block, flags=re.DOTALL)
        if not (m1 and m2 and mv):
            raise ValueError("Missing index_1, index_2, or values in LUT block")

        idx1_str = _normalize_csv_string(m1.group(1))
        idx2_str = _normalize_csv_string(m2.group(1))
        row_strs = re.findall(r'"([^"]+)"', mv.group(1), flags=re.DOTALL)
        if not row_strs:
            raise ValueError("No quoted rows inside values(...)")

        values_str_rows = [_normalize_csv_string(r) for r in row_strs]
        idx1   = _csv_to_floats(idx1_str)
        idx2   = _csv_to_floats(idx2_str)
        values = [[float(t) for t in row.split(",")] for row in values_str_rows]

        if len(values) != len(idx1):
            raise ValueError(f"LUT rows ({len(values)}) != len(index_1) ({len(idx1)})")
        for r in values:
            if len(r) != len(idx2):
                raise ValueError(f"LUT cols ({len(r)}) != len(index_2) ({len(idx2)})")

        return LUT(index_1_str=idx1_str, index_2_str=idx2_str,
                   values_str_rows=values_str_rows,
                   index_1=idx1, index_2=idx2, values=values)

    def extract_table(cell_block: str, table_name: str) -> Optional[LUT]:
        m = re.search(rf"\b{re.escape(table_name)}\b\s*\(", cell_block)
        if not m:
            return None
        ob = cell_block.find("{", m.end())
        if ob == -1:
            return None
        cb = _find_matching_brace(cell_block, ob)
        return parse_lut(cell_block[ob + 1: cb])

    def extract_input_cap(cell_block: str) -> float:
        """Return input capacitance (fF): check input pin blocks first, then cell-level attr."""
        # Try per-pin capacitance inside an input pin block
        for pm in re.finditer(r'pin\s*\([^)]+\)\s*\{', cell_block):
            ob = cell_block.find("{", pm.end())
            if ob == -1:
                continue
            cb = _find_matching_brace(cell_block, ob)
            pin_block = cell_block[ob + 1: cb]
            if re.search(r'direction\s*:\s*input', pin_block):
                cm = re.search(r'capacitance\s*:\s*([\d.e+\-]+)', pin_block)
                if cm:
                    return float(cm.group(1))
        # Fall back to cell-level  capacitance attribute
        cm = re.search(r'\bcapacitance\s*:\s*([\d.e+\-]+)', cell_block)
        if cm:
            return float(cm.group(1))
        return 0.0

    cells: Dict[str, NLDMCell] = {}
    pos = 0
    while True:
        m = re.search(r"\bcell\s*\(\s*([A-Za-z0-9_]+)\s*\)\s*\{", text[pos:])
        if not m:
            break
        cell_name  = m.group(1)
        brace_open = pos + m.end() - 1
        brace_close = _find_matching_brace(text, brace_open)
        cell_block  = text[brace_open + 1: brace_close]

        cells[cell_name] = NLDMCell(
            name        = cell_name,
            cell_delay  = extract_table(cell_block, "cell_delay"),
            output_slew = extract_table(cell_block, "output_slew"),
            input_cap   = extract_input_cap(cell_block),
        )
        pos = brace_close + 1

    if not cells:
        raise ValueError("No cells found in .lib file")
    return cells


def write_delay_lut(nldm: Dict[str, NLDMCell], out_path: Path) -> None:
    with out_path.open("w") as w:
        for cell_name, cell in nldm.items():
            if cell.cell_delay is None:
                continue
            lut  = cell.cell_delay
            rows = lut.values_str_rows
            w.write(f"cell: {cell_name}\n")
            w.write(f"input slews: {lut.index_1_str}\n")
            w.write(f"load cap: {lut.index_2_str}\n")
            w.write(f"delays: {rows[0]};\n")
            for row in rows[1:]:
                w.write(f"{row};\n")


def write_slew_lut(nldm: Dict[str, NLDMCell], out_path: Path) -> None:
    with out_path.open("w") as w:
        for cell_name, cell in nldm.items():
            if cell.output_slew is None:
                continue
            lut  = cell.output_slew
            rows = lut.values_str_rows
            w.write(f"cell: {cell_name}\n")
            w.write(f"input slews: {lut.index_1_str}\n")
            w.write(f"load cap: {lut.index_2_str}\n")
            w.write(f"slews: {rows[0]};\n")
            for row in rows[1:]:
                w.write(f"{row};\n")


# ---------------------------------------------------------------------------
# Phase-2: Static Timing Analysis
# ---------------------------------------------------------------------------

def run_sta(primary_inputs, primary_outputs, nodes, gate_order,
            nldm: Dict[str, NLDMCell], out_path: Path) -> None:

    pi_set = set(primary_inputs)

    # 1. Find inverter capacitance (for final-stage load)
    inv_cap = _find_inv_cap(nldm)

    # 2. Manual topological sort (Kahn's algorithm — no graph libraries)
    topo_order = _topo_sort(nodes, gate_order)

    # 3. Assign load capacitances
    _compute_cload(topo_order, nodes, nldm, inv_cap)

    # 4. Forward traversal → arrival times & output slews
    circuit_delay_ns = _forward_traversal(topo_order, nodes, pi_set, nldm)

    # 5. Backward traversal → required arrival times & gate slacks (ps)
    _backward_traversal(topo_order, nodes, circuit_delay_ns)

    # 6. Primary-input slacks (ps)
    pi_slacks = _compute_pi_slacks(primary_inputs, nodes, gate_order)

    # 7. Critical path
    critical_path = _find_critical_path(primary_outputs, nodes, pi_slacks)

    # 8. Write output
    _write_traversal(primary_inputs, primary_outputs, nodes, gate_order,
                     pi_slacks, circuit_delay_ns, critical_path, out_path)


# ---- helpers ---------------------------------------------------------------

def _find_inv_cap(nldm: Dict[str, NLDMCell]) -> float:
    """Return the input capacitance of the inverter/NOT cell."""
    for name, cell in nldm.items():
        if "inv" in name.lower() or name.lower().startswith("not"):
            if cell.input_cap > 0:
                return cell.input_cap
    # Fallback: smallest positive input cap
    caps = [c.input_cap for c in nldm.values() if c.input_cap > 0]
    return min(caps) if caps else 1.0


def _find_cell(gate_type: str, nldm: Dict[str, NLDMCell]) -> Optional[NLDMCell]:
    """
    Find the best-matching NLDM cell for a bench gate_type.
    Handles: NAND→NAND2_X1, NOR→NOR2_X1, AND→AND2_X1, OR→OR2_X1, NOT/INV→INV_X1, etc.
    Priority: (1) exact match, (2) starts-with, with NOT/INV aliased together.
    """
    gt_lo = gate_type.lower()

    # Build alias set: treat NOT and INV as equivalent
    aliases = {gt_lo}
    if gt_lo in ("not", "inv", "inverter"):
        aliases.update(("not", "inv", "inverter"))
    if gt_lo in ("buf", "buff", "buffer"):
        aliases.update(("buf", "buff", "buffer"))

    # 1. Exact case-insensitive match
    for name, c in nldm.items():
        if name.lower() in aliases:
            return c

    # 2. Cell name starts with one of the aliases (handles "NAND2_X1" for "NAND")
    #    Sort by name length ascending so we prefer the shortest (simplest) match
    for name, c in sorted(nldm.items(), key=lambda x: len(x[0])):
        nl = name.lower()
        for alias in aliases:
            if nl.startswith(alias):
                return c

    return None


def _lut_interp(lut: LUT, tau_ns: float, C_fF: float) -> float:
    """
    Bilinear interpolation on a 2-D LUT.
    index_1 → input slew (ns), index_2 → load cap (fF).
    Values outside the table range are clamped to the boundary.
    """
    tau_vals = lut.index_1
    C_vals   = lut.index_2
    V        = lut.values

    # Clamp query to table range
    tau_eff = max(tau_vals[0], min(tau_vals[-1], tau_ns))
    C_eff   = max(C_vals[0],   min(C_vals[-1],   C_fF))

    def find_bracket(arr: List[float], x: float):
        if x >= arr[-1]:
            return len(arr) - 2, len(arr) - 1
        for i in range(len(arr) - 1):
            if arr[i] <= x < arr[i + 1]:
                return i, i + 1
        return 0, 1

    i1, i2 = find_bracket(tau_vals, tau_eff)
    j1, j2 = find_bracket(C_vals,   C_eff)

    tau1, tau2 = tau_vals[i1], tau_vals[i2]
    C1,   C2   = C_vals[j1],   C_vals[j2]
    v11, v12   = V[i1][j1],    V[i1][j2]
    v21, v22   = V[i2][j1],    V[i2][j2]

    d_tau = tau2 - tau1
    d_C   = C2   - C1

    if d_tau == 0.0 and d_C == 0.0:
        return v11
    if d_tau == 0.0:
        return v11 + (v12 - v11) * (C_eff - C1) / d_C
    if d_C == 0.0:
        return v11 + (v21 - v11) * (tau_eff - tau1) / d_tau

    v = (v11 * (C2   - C_eff) * (tau2 - tau_eff)
       + v12 * (C_eff - C1)   * (tau2 - tau_eff)
       + v21 * (C2   - C_eff) * (tau_eff - tau1)
       + v22 * (C_eff - C1)   * (tau_eff - tau1))
    return v / (d_C * d_tau)


def _topo_sort(nodes: Dict[str, Node], gate_order: List[str]) -> List[str]:
    """
    Kahn's topological sort — implemented manually without any library.
    Returns gate nets in topological (evaluation) order.
    """
    in_deg: Dict[str, int] = {}
    for net in gate_order:
        in_deg[net] = sum(1 for fi in nodes[net].fanin if isinstance(fi, Node))

    queue  = [net for net in gate_order if in_deg[net] == 0]
    result: List[str] = []

    while queue:
        net = queue.pop(0)
        result.append(net)
        for fo in nodes[net].fanout:
            if isinstance(fo, Node):
                in_deg[fo.out_net] -= 1
                if in_deg[fo.out_net] == 0:
                    queue.append(fo.out_net)

    if len(result) != len(gate_order):
        raise ValueError("Cycle detected in netlist — topological sort failed")
    return result


def _compute_cload(topo_order: List[str], nodes: Dict[str, Node],
                   nldm: Dict[str, NLDMCell], inv_cap: float) -> None:
    """
    Assign Cload (fF) to every gate:
      • Gates that only drive primary outputs → 4 × inv_cap (per PO output).
      • All other gates → sum of input caps of their gate fanouts,
        plus 4 × inv_cap for each PO fanout if mixed.
    """
    for out_net in topo_order:
        gate = nodes[out_net]

        gate_fanouts = [fo for fo in gate.fanout if isinstance(fo, Node)]
        po_fanouts   = [fo for fo in gate.fanout if isinstance(fo, str)
                        and fo.startswith("OUTPUT-")]

        if po_fanouts and not gate_fanouts:
            # Final-stage gate: only drives primary outputs
            gate.Cload = 4.0 * inv_cap * len(po_fanouts)
        else:
            cload = 0.0
            for fo in gate_fanouts:
                cell = _find_cell(fo.gate_type, nldm)
                if cell:
                    cload += cell.input_cap
            # Mixed: also add PO load if any PO fanouts
            cload += 4.0 * inv_cap * len(po_fanouts)
            gate.Cload = cload


def _gate_delay_slew(gate: Node, nldm: Dict[str, NLDMCell]):
    """
    Return (delays, slews) in ns for each input path of gate.
    For n > 2 inputs: scale LUT result by n/2  (spec assumption).
    """
    cell = _find_cell(gate.gate_type, nldm)
    if cell is None:
        raise ValueError(f"No NLDM cell found for gate type '{gate.gate_type}'")
    if cell.cell_delay is None or cell.output_slew is None:
        raise ValueError(f"Cell '{cell.name}' missing delay or slew LUT")

    n     = len(gate.fanin)
    scale = (n / 2.0) if n > 2 else 1.0   # spec: scale only for n > 2

    delays: List[float] = []
    slews:  List[float] = []
    for i in range(n):
        tau = gate.Tau_in[i]
        C   = gate.Cload
        delays.append(_lut_interp(cell.cell_delay,  tau, C) * scale)
        slews.append( _lut_interp(cell.output_slew, tau, C) * scale)

    return delays, slews


def _forward_traversal(topo_order: List[str], nodes: Dict[str, Node],
                       pi_set: set, nldm: Dict[str, NLDMCell]) -> float:
    """
    Forward STA pass.
    Populates: Tau_in, inp_arrival, cell_delays, cell_slews,
               outp_arrival, max_out_arrival, Tau_out for every gate.
    Returns circuit delay (ns) = max arrival among final-stage outputs.
    """
    TAU_PI = 0.002   # 2 ps expressed in ns
    ARR_PI = 0.0     # primary inputs arrive at time 0

    for out_net in topo_order:
        gate = nodes[out_net]
        gate.Tau_in      = []
        gate.inp_arrival = []

        for fi in gate.fanin:
            if isinstance(fi, str):          # PRIMARY INPUT
                gate.Tau_in.append(TAU_PI)
                gate.inp_arrival.append(ARR_PI)
            else:                            # gate output
                gate.Tau_in.append(fi.Tau_out)
                gate.inp_arrival.append(fi.max_out_arrival)

        delays, slews      = _gate_delay_slew(gate, nldm)
        gate.cell_delays   = delays
        gate.cell_slews    = slews
        gate.outp_arrival  = [gate.inp_arrival[i] + delays[i] for i in range(len(delays))]

        # max: a_out = max_i(a_i + d_i)
        max_idx = 0
        for i in range(1, len(gate.outp_arrival)):
            if gate.outp_arrival[i] > gate.outp_arrival[max_idx]:
                max_idx = i

        gate.max_out_arrival = gate.outp_arrival[max_idx]
        gate.Tau_out         = slews[max_idx]   # τ_out = argmax slew

    # Circuit delay = max arrival at all primary outputs
    circuit_delay = 0.0
    for out_net in topo_order:
        gate = nodes[out_net]
        has_po = any(isinstance(fo, str) and fo.startswith("OUTPUT-")
                     for fo in gate.fanout)
        if has_po:
            circuit_delay = max(circuit_delay, gate.max_out_arrival)

    return circuit_delay


def _backward_traversal(topo_order: List[str], nodes: Dict[str, Node],
                        circuit_delay_ns: float) -> None:
    """
    Backward STA pass.
    Populates: req_arrival (ns) and slack (ps) for every gate.
    Required arrival time = 1.1 × circuit_delay at every primary output.
    """
    req_time = 1.1 * circuit_delay_ns

    for out_net in reversed(topo_order):
        gate = nodes[out_net]

        gate_fanouts = [fo for fo in gate.fanout if isinstance(fo, Node)]
        has_po       = any(isinstance(fo, str) and fo.startswith("OUTPUT-")
                           for fo in gate.fanout)

        if has_po and not gate_fanouts:
            # Final-stage gate: required time set directly
            gate.req_arrival = req_time

        else:
            # req_arrival = min over fanout gates of
            #               (fanout.req_arrival − fanout.cell_delays[idx connecting to this gate])
            min_req: Optional[float] = None

            for fo in gate_fanouts:
                for i, fi in enumerate(fo.fanin):
                    if isinstance(fi, Node) and fi.out_net == gate.out_net:
                        r = fo.req_arrival - fo.cell_delays[i]
                        if min_req is None or r < min_req:
                            min_req = r

            # If also drives a PO directly, that boundary must be respected too
            if has_po:
                if min_req is None or req_time < min_req:
                    min_req = req_time

            gate.req_arrival = min_req if min_req is not None else req_time

        gate.slack = (gate.req_arrival - gate.max_out_arrival) * 1000.0  # → ps


def _compute_pi_slacks(primary_inputs: List[str], nodes: Dict[str, Node],
                       gate_order: List[str]) -> Dict[str, float]:
    """
    Compute slack (ps) for each primary input.
    Slack = min over all fanout gates of (gate.req_arrival − gate.cell_delays[i]) × 1000,
    since arrival time at every PI is 0.
    """
    pi_slacks: Dict[str, float] = {}

    for pi_net in primary_inputs:
        pi_label = f"INPUT-{pi_net}"
        req_list: List[float] = []

        for out_net in gate_order:
            node = nodes[out_net]
            for i, fi in enumerate(node.fanin):
                if isinstance(fi, str) and fi == pi_label:
                    req_at_pi_ps = (node.req_arrival - node.cell_delays[i]) * 1000.0
                    req_list.append(req_at_pi_ps)

        pi_slacks[pi_label] = min(req_list) if req_list else 0.0

    return pi_slacks


def _find_critical_path(primary_outputs: List[str], nodes: Dict[str, Node],
                        pi_slacks: Dict[str, float]) -> List[str]:
    """
    Trace critical path from the PO with minimum slack back to the PI with minimum slack.
    Returns list of labels from PI to PO.
    """
    # Find PO with minimum slack (PO slack == driving gate slack)
    min_slack: Optional[float] = None
    start_net: Optional[str]   = None
    for po_net in primary_outputs:
        if po_net in nodes:
            s = nodes[po_net].slack
            if min_slack is None or s < min_slack:
                min_slack = s
                start_net = po_net

    if start_net is None:
        return []

    # Trace backward, collecting labels in reverse order
    rev_path: List[str] = [f"OUTPUT-{start_net}"]
    current = nodes[start_net]

    while True:
        rev_path.append(current.label())

        # Collect all fanin candidates with their slacks
        candidates = []
        for fi in current.fanin:
            if isinstance(fi, Node):
                candidates.append((fi.slack, fi.label(), "gate", fi))
            else:
                s = pi_slacks.get(fi, 0.0)
                candidates.append((s, fi, "pi", fi))

        if not candidates:
            break

        best = min(candidates, key=lambda x: x[0])
        _, _, typ, fi = best

        if typ == "pi":
            rev_path.append(fi)   # "INPUT-x"
            break
        else:
            current = fi          # type: Node

    rev_path.reverse()
    return rev_path


def _write_traversal(primary_inputs, primary_outputs, nodes, gate_order,
                     pi_slacks, circuit_delay_ns, critical_path, out_path: Path) -> None:
    delay_ps = circuit_delay_ns * 1000.0

    with out_path.open("w") as w:
        w.write(f"Circuit delay: {delay_ps:.6g} ps\n")
        w.write("\nGate slacks:\n")

        # Primary inputs
        for pi_net in primary_inputs:
            label = f"INPUT-{pi_net}"
            w.write(f"{label}: {pi_slacks.get(label, 0.0):.6g} ps\n")

        # Primary outputs  (slack == slack of driving gate)
        for po_net in primary_outputs:
            label = f"OUTPUT-{po_net}"
            slack = nodes[po_net].slack if po_net in nodes else 0.0
            w.write(f"{label}: {slack:.6g} ps\n")

        # Internal gates (in original bench order)
        for out_net in gate_order:
            node = nodes[out_net]
            w.write(f"{node.label()}: {node.slack:.6g} ps\n")

        w.write("\nCritical path:\n")
        w.write(",".join(critical_path) + "\n")


if __name__ == "__main__":
    main()