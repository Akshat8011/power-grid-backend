import os
import copy
import logging
import numpy as np
import pandapower as pp
import pandapower.shortcircuit as sc
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# ══════════════════════════════════════════════════
#  LOGGING CONFIGURATION
# ══════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("power-grid-physics")

# ══════════════════════════════════════════════════
#  FASTAPI SETUP
# ══════════════════════════════════════════════════
app = FastAPI(
    title="Power Grid Digital Twin — Physics Engine",
    version="2.0.0",
    description="SC-Hardened pandapower API with weather, economics, and deep telemetry.",
)

# Global CORS — Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════
#  PANDAPOWER GRID DEFINITION (SC-HARDENED)
# ══════════════════════════════════════════════════
def create_city_grid() -> pp.pandapowerNet:
    net = pp.create_empty_network(name="CityGrid_v2", f_hz=50.0)

    # 1. BUSES
    bus_hv    = pp.create_bus(net, vn_kv=110, name="HV Bus (110kV)")              # 0
    bus_mv1   = pp.create_bus(net, vn_kv=33,  name="MV Primary (33kV)")           # 1
    bus_mv2   = pp.create_bus(net, vn_kv=11,  name="MV Substation (11kV)")        # 2
    bus_res   = pp.create_bus(net, vn_kv=11,  name="Residential District (11kV)") # 3
    bus_com   = pp.create_bus(net, vn_kv=11,  name="Commercial District (11kV)")  # 4
    bus_ind   = pp.create_bus(net, vn_kv=11,  name="Industrial District (11kV)")  # 5

    # 2. EXTERNAL GRID (Full IEC 60909 Parameters)
    pp.create_ext_grid(
        net, bus=bus_hv, vm_pu=1.02, name="Utility Grid",
        s_sc_max_mva=1000, rx_max=0.1,
        s_sc_min_mva=800,  rx_min=0.1,
        x0x_max=0.1, r0x0_max=0.1,
    )

    # 3. TRANSFORMERS (Standard Types + Voltage Override)
    t1 = pp.create_transformer(net, hv_bus=bus_hv, lv_bus=bus_mv1,
                          std_type="63 MVA 110/20 kV", name="T1")
    t2 = pp.create_transformer(net, hv_bus=bus_mv1, lv_bus=bus_mv2,
                          std_type="25 MVA 110/20 kV", name="T2")

    net.trafo.at[t1, "vn_hv_kv"] = 110.0
    net.trafo.at[t1, "vn_lv_kv"] = 33.0
    net.trafo.at[t2, "vn_hv_kv"] = 33.0
    net.trafo.at[t2, "vn_lv_kv"] = 11.0

    # GLOBAL SEQUENCE SYNC — fills all mandatory IEC 60909 columns
    net.trafo["vector_group"] = "Dyn"
    net.trafo["vk0_percent"] = net.trafo["vk_percent"]
    net.trafo["vkr0_percent"] = net.trafo["vkr_percent"]
    net.trafo["mag0_percent"] = 100.0
    net.trafo["mag0_rx"] = 0.4
    net.trafo["si0_hv_partial"] = 1.0
    net.trafo["si0_lv_partial"] = 1.0

    # 4. LINES (Enriched with zero-sequence)
    r1 = 0.642; x1 = 0.083; c1 = 210
    r0 = r1 * 4; x0 = x1 * 4; c0 = c1

    pp.create_line_from_parameters(
        net, from_bus=bus_mv2, to_bus=bus_res, length_km=5.0,
        r_ohm_per_km=r1, x_ohm_per_km=x1, c_nf_per_km=c1, max_i_ka=0.21,
        r0_ohm_per_km=r0, x0_ohm_per_km=x0, c0_nf_per_km=c0,
        name="Feeder → Residential"
    )
    pp.create_line_from_parameters(
        net, from_bus=bus_mv2, to_bus=bus_com, length_km=4.0,
        r_ohm_per_km=r1, x_ohm_per_km=x1, c_nf_per_km=c1, max_i_ka=0.21,
        r0_ohm_per_km=r0, x0_ohm_per_km=x0, c0_nf_per_km=c0,
        name="Feeder → Commercial"
    )
    pp.create_line_from_parameters(
        net, from_bus=bus_mv2, to_bus=bus_ind, length_km=8.0,
        r_ohm_per_km=r1, x_ohm_per_km=x1, c_nf_per_km=c1, max_i_ka=0.21,
        r0_ohm_per_km=r0, x0_ohm_per_km=x0, c0_nf_per_km=c0,
        name="Feeder → Industrial"
    )

    # Sync line zero-sequence
    if "r0_ohm_per_km" not in net.line.columns:
        net.line["r0_ohm_per_km"] = net.line["r_ohm_per_km"] * 4
        net.line["x0_ohm_per_km"] = net.line["x_ohm_per_km"] * 4
        net.line["c0_nf_per_km"] = net.line["c_nf_per_km"]

    # 5. LOADS
    pp.create_load(net, bus=bus_res, p_mw=0.096, q_mvar=0.031, name="Residential Load")
    pp.create_load(net, bus=bus_com, p_mw=0.150, q_mvar=0.064, name="Commercial Load")
    pp.create_load(net, bus=bus_ind, p_mw=0.950, q_mvar=0.588, name="Industrial Load")

    # 6. SOLAR
    pp.create_sgen(net, bus=bus_res, p_mw=0.02, sn_mva=0.03, k=1.0, name="Res Solar")
    pp.create_sgen(net, bus=bus_com, p_mw=0.02, sn_mva=0.03, k=1.0, name="Com Solar")
    pp.create_sgen(net, bus=bus_ind, p_mw=0.05, sn_mva=0.07, k=1.0, name="Ind Solar")

    if len(net.sgen) > 0:
        net.sgen["k"] = 1.0

    return net

MASTER_NET = create_city_grid()

# ══════════════════════════════════════════════════
#  GRID TOPOLOGY MAP (for cascade analysis)
# ══════════════════════════════════════════════════
# Adjacency: bus_from -> bus_to via lines and transformers
GRID_ADJACENCY = {
    0: [1],     # HV -> MV1 via T1
    1: [2],     # MV1 -> MV2 via T2
    2: [3, 4, 5],  # MV2 -> Residential, Commercial, Industrial
    3: [], 4: [], 5: []
}

def find_dead_nodes(faulted_bus: int) -> list:
    """BFS traversal to find all buses isolated downstream of the faulted bus."""
    dead = set()
    queue = [faulted_bus]
    while queue:
        bus = queue.pop(0)
        if bus in dead:
            continue
        dead.add(bus)
        for child in GRID_ADJACENCY.get(bus, []):
            if child not in dead:
                queue.append(child)
    return sorted(list(dead))

def find_faulted_lines(faulted_bus: int) -> list:
    """Find all line indices that connect to the faulted bus or its children."""
    dead_buses = find_dead_nodes(faulted_bus)
    faulted_lines = []
    net = MASTER_NET
    for idx in net.line.index:
        from_bus = int(net.line.at[idx, "from_bus"])
        to_bus = int(net.line.at[idx, "to_bus"])
        if from_bus in dead_buses or to_bus in dead_buses:
            faulted_lines.append(int(idx))
    return faulted_lines


# ══════════════════════════════════════════════════
#  API MODELS
# ══════════════════════════════════════════════════
class LoadFlowRequest(BaseModel):
    residential_kw: float = 96.0
    commercial_kw: float = 150.0
    industrial_kw: float = 950.0
    residential_solar_kw: float = 20.0
    commercial_solar_kw: float = 20.0
    industrial_solar_kw: float = 50.0
    # Weather multipliers
    load_multiplier: float = 1.0
    solar_multiplier: float = 1.0

class FaultRequest(BaseModel):
    bus_index: int = 2
    fault_type: str = "3ph"


# ══════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "ok", "version": "2.0.0"}

@app.post("/api/loadflow")
@app.post("/api/loadflow/")
def run_load_flow(req: LoadFlowRequest):
    net = copy.deepcopy(MASTER_NET)

    def kw_to_mw(kw): return round(kw / 1000.0, 6)

    # Apply weather multipliers
    res_kw = req.residential_kw * req.load_multiplier
    com_kw = req.commercial_kw * req.load_multiplier
    ind_kw = req.industrial_kw * req.load_multiplier

    res_solar = req.residential_solar_kw * req.solar_multiplier
    com_solar = req.commercial_solar_kw * req.solar_multiplier
    ind_solar = req.industrial_solar_kw * req.solar_multiplier

    # Update loads
    net.load.at[0, "p_mw"] = kw_to_mw(res_kw)
    net.load.at[1, "p_mw"] = kw_to_mw(com_kw)
    net.load.at[2, "p_mw"] = kw_to_mw(ind_kw)

    # Update sgen
    net.sgen.at[0, "p_mw"] = kw_to_mw(res_solar)
    net.sgen.at[1, "p_mw"] = kw_to_mw(com_solar)
    net.sgen.at[2, "p_mw"] = kw_to_mw(ind_solar)

    try:
        pp.runpp(net)

        # Per-line results
        lines = []
        for idx, row in net.res_line.iterrows():
            lines.append({
                "index": int(idx),
                "loading_percent": round(float(row["loading_percent"]), 2),
                "p_from_mw": round(float(row["p_from_mw"]), 6),
                "p_to_mw": round(float(row.get("p_to_mw", 0)), 6),
                "pl_mw": round(float(row["pl_mw"]), 6),
                "i_ka": round(float(row["i_ka"]), 4),
            })

        # Per-bus results
        buses = []
        for idx, row in net.res_bus.iterrows():
            buses.append({
                "index": int(idx),
                "vm_pu": round(float(row["vm_pu"]), 4),
                "va_degree": round(float(row["va_degree"]), 2),
                "p_mw": round(float(row.get("p_mw", 0)), 6),
                "q_mvar": round(float(row.get("q_mvar", 0)), 6),
            })

        # Per-trafo results
        trafos = []
        for idx, row in net.res_trafo.iterrows():
            trafos.append({
                "index": int(idx),
                "loading_percent": round(float(row["loading_percent"]), 2),
                "pl_mw": round(float(row["pl_mw"]), 6),
                "p_hv_mw": round(float(row["p_hv_mw"]), 6),
                "p_lv_mw": round(float(row["p_lv_mw"]), 6),
            })

        total_gen = float(net.res_ext_grid["p_mw"].sum()) * 1000
        total_demand = res_kw + com_kw + ind_kw
        total_solar = res_solar + com_solar + ind_solar
        total_line_loss = float(net.res_line["pl_mw"].sum()) * 1000
        total_trafo_loss = float(net.res_trafo["pl_mw"].sum()) * 1000
        total_loss = total_line_loss + total_trafo_loss

        return {
            "converged": True,
            "summary": {
                "total_generation_kw": round(total_gen, 2),
                "total_demand_kw": round(total_demand, 2),
                "total_solar_kw": round(total_solar, 2),
                "total_loss_kw": round(total_loss, 2),
                "total_line_loss_kw": round(total_line_loss, 2),
                "total_trafo_loss_kw": round(total_trafo_loss, 2),
                "loss_percent": round((total_loss / max(total_gen, 0.01)) * 100, 2),
                "curtailed_kw": 0,
            },
            "lines": lines,
            "buses": buses,
            "transformers": trafos,
        }
    except Exception as e:
        logger.error("LOADFLOW ERROR: %s", str(e))
        return {"converged": False, "error": str(e)}


@app.post("/api/fault")
@app.post("/api/fault/")
def run_fault_analysis(req: FaultRequest):
    net = copy.deepcopy(MASTER_NET)

    if req.bus_index < 0 or req.bus_index >= len(net.bus):
        raise HTTPException(status_code=400, detail="Invalid bus index")

    try:
        if req.fault_type == "1ph":
            sc.calc_sc(net, bus=req.bus_index, fault=req.fault_type)
        else:
            sc.calc_sc(net, bus=req.bus_index, fault=req.fault_type, ip=True, ith=True)

        res = net.res_bus_sc.loc[req.bus_index]

        # Deep cascade analysis
        dead_nodes = find_dead_nodes(req.bus_index)
        faulted_lines = find_faulted_lines(req.bus_index)

        # Map dead nodes to district names and load values
        dead_node_details = []
        for bus_idx in dead_nodes:
            name = net.bus.at[bus_idx, "name"]
            load_mw = 0
            for li in net.load.index:
                if int(net.load.at[li, "bus"]) == bus_idx:
                    load_mw += float(net.load.at[li, "p_mw"])
            dead_node_details.append({
                "bus_index": bus_idx,
                "name": name,
                "lost_load_kw": round(load_mw * 1000, 2),
            })

        total_lost_kw = sum(d["lost_load_kw"] for d in dead_node_details)

        return {
            "success": True,
            "faulted_bus_index": req.bus_index,
            "faulted_bus_name": net.bus.at[req.bus_index, "name"],
            "fault_type": req.fault_type,
            "fault_current": {
                "ikss_ka": round(float(res["ikss_ka"]), 4),
                "ip_ka": round(float(res.get("ip_ka", 0)), 4),
            },
            "breaker_trip": True,
            "dead_nodes": dead_nodes,
            "dead_node_details": dead_node_details,
            "faulted_line_indices": faulted_lines,
            "total_lost_load_kw": round(total_lost_kw, 2),
            "downstream_buses_offline": dead_nodes,
            "blackout_zones": [net.bus.at[i, "name"] for i in dead_nodes if i >= 2],
            "faulted_line_index": faulted_lines[0] if faulted_lines else 0,
        }
    except Exception as e:
        logger.error("FAULT ERROR: %s", str(e))
        return {
            "success": False,
            "error": str(e),
            "detail": f"Physics Error: {str(e)}"
        }


# ENTRYPOINT
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
