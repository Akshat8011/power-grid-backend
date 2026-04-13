"""
Power Grid Digital Twin — Backend Engine
=========================================
Uses pandapower for Newton-Raphson load flow & IEC 60909 short-circuit analysis.
Deployed on Railway, consumed by React Three Fiber frontend on Vercel.
"""

import os
import copy
import logging
from typing import Dict, Optional

import pandapower as pp
import pandapower.shortcircuit as sc
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ══════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grid-engine")

# ══════════════════════════════════════════════════
#  FASTAPI APP + CORS
# ══════════════════════════════════════════════════
app = FastAPI(
    title="Power Grid Digital Twin — Physics Engine",
    version="1.0.0",
    description="pandapower-based load flow & fault analysis API for the 3D city grid.",
)

# Allow both local dev and production Vercel domain
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://load-analysis.vercel.app",
    os.getenv("FRONTEND_URL", ""),        # Railway env var override
]
# Filter out empty strings
ALLOWED_ORIGINS = [o for o in ALLOWED_ORIGINS if o]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════
#  PANDAPOWER GRID DEFINITION
# ══════════════════════════════════════════════════
def create_city_grid() -> pp.pandapowerNet:
    """
    Build a realistic city distribution grid:

    External Grid (Slack)
        │  110 kV
        ▼
    Bus 0 ── HV Bus (110 kV)
        │
      [Transformer 110/33 kV]
        │
    Bus 1 ── MV Primary Bus (33 kV)
        │
      [Transformer 33/11 kV]
        │
    Bus 2 ── MV Secondary Bus (11 kV)   ← Central City Substation
        │
        ├── Line → Bus 3 (Residential District)
        │             └ Load: 24 houses × ~4 kW = 96 kW
        │
        ├── Line → Bus 4 (Commercial District)
        │             └ Load: Supermarket 100kW + School 50kW = 150 kW
        │
        └── Line → Bus 5 (Industrial/Medical District)
                      └ Load: Hospital 500kW + Factory 450kW = 950 kW
    """
    net = pp.create_empty_network(name="CityGrid_v1", f_hz=50.0)

    # ── BUSES ──────────────────────────────────────
    bus_hv    = pp.create_bus(net, vn_kv=110, name="HV Bus (110kV)")              # 0
    bus_mv1   = pp.create_bus(net, vn_kv=33,  name="MV Primary (33kV)")           # 1
    bus_mv2   = pp.create_bus(net, vn_kv=11,  name="MV Substation (11kV)")        # 2
    bus_res   = pp.create_bus(net, vn_kv=11,  name="Residential District (11kV)") # 3
    bus_com   = pp.create_bus(net, vn_kv=11,  name="Commercial District (11kV)")  # 4
    bus_ind   = pp.create_bus(net, vn_kv=11,  name="Industrial District (11kV)")  # 5

    # ── EXTERNAL GRID (Slack / Infinite Bus) ──────
    pp.create_ext_grid(
        net, bus=bus_hv, vm_pu=1.02, name="Utility Grid Connection",
        s_sc_max_mva=1000, rx_max=0.1,   # short-circuit params for IEC 60909
        s_sc_min_mva=800,  rx_min=0.1,
    )

    # ── TRANSFORMERS ──────────────────────────────
    # 110 kV → 33 kV  (Step-Down #1)
    pp.create_transformer(
        net, hv_bus=bus_hv, lv_bus=bus_mv1,
        std_type="63 MVA 110/20 kV", name="Step-Down 110→33kV"
    )
    # 33 kV → 11 kV   (Step-Down #2, the City Substation)
    pp.create_transformer(
        net, hv_bus=bus_mv1, lv_bus=bus_mv2,
        std_type="25 MVA 110/20 kV", name="Substation Step-Down 33→11kV"
    )

    # ── DISTRIBUTION LINES (11 kV feeders) ────────
    # Residential feeder: 5 km
    pp.create_line(
        net, from_bus=bus_mv2, to_bus=bus_res,
        length_km=5.0, std_type="NAYY 4x50 SE",
        name="Feeder → Residential"
    )
    # Commercial feeder: 4 km
    pp.create_line(
        net, from_bus=bus_mv2, to_bus=bus_com,
        length_km=4.0, std_type="NAYY 4x50 SE",
        name="Feeder → Commercial"
    )
    # Industrial feeder: 8 km
    pp.create_line(
        net, from_bus=bus_mv2, to_bus=bus_ind,
        length_km=8.0, std_type="NAYY 4x50 SE",
        name="Feeder → Industrial"
    )

    # ── LOADS ─────────────────────────────────────
    # Residential: 24 houses × 4 kW avg = 96 kW active, pf ≈ 0.95
    pp.create_load(
        net, bus=bus_res, p_mw=0.096, q_mvar=0.031,
        name="Residential Load"
    )
    # Commercial: Supermarket (100 kW) + School (50 kW) = 150 kW, pf ≈ 0.92
    pp.create_load(
        net, bus=bus_com, p_mw=0.150, q_mvar=0.064,
        name="Commercial Load"
    )
    # Industrial: Hospital (500 kW) + Factory (450 kW) = 950 kW, pf ≈ 0.85
    pp.create_load(
        net, bus=bus_ind, p_mw=0.950, q_mvar=0.588,
        name="Industrial Load"
    )

    # ── STATIC GENERATORS (Rooftop Solar DERs) ───
    # Some residential houses have solar panels (total ~20 kW)
    pp.create_sgen(
        net, bus=bus_res, p_mw=0.020, q_mvar=0.0,
        name="Residential Rooftop Solar", type="PV"
    )
    # Supermarket rooftop solar (20 kW)
    pp.create_sgen(
        net, bus=bus_com, p_mw=0.020, q_mvar=0.0,
        name="Commercial Rooftop Solar", type="PV"
    )
    # Hospital backup solar (50 kW)
    pp.create_sgen(
        net, bus=bus_ind, p_mw=0.050, q_mvar=0.0,
        name="Hospital Solar Farm", type="PV"
    )

    logger.info("✅ pandapower grid created: %d buses, %d lines, %d loads",
                len(net.bus), len(net.line), len(net.load))
    return net


# Create the master grid once at startup
MASTER_NET = create_city_grid()


# ══════════════════════════════════════════════════
#  PYDANTIC REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════

class LoadFlowRequest(BaseModel):
    """
    Dynamic load values from frontend appliance toggles.
    All values in kW. The backend converts to MW for pandapower.
    """
    residential_kw: float = 96.0
    commercial_kw: float = 150.0
    industrial_kw: float = 950.0
    # Optional solar generation overrides (kW)
    residential_solar_kw: float = 20.0
    commercial_solar_kw: float = 20.0
    industrial_solar_kw: float = 50.0


class FaultRequest(BaseModel):
    """Fault simulation parameters."""
    bus_index: int = 2        # Default: fault at the City Substation bus
    fault_type: str = "3ph"   # '3ph' or '2ph' or '1ph'


# ══════════════════════════════════════════════════
#  HEALTH CHECK
# ══════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "ok"}



# ══════════════════════════════════════════════════
#  PHASE 3: LOAD FLOW ANALYSIS
# ══════════════════════════════════════════════════

@app.post("/api/loadflow")
@app.post("/api/loadflow/")
def run_load_flow(req: LoadFlowRequest):
    """
    Accepts dynamic load values from the 3D frontend, runs Newton-Raphson
    power flow, and returns bus voltages, line loadings, and total losses.
    """
    # Deep copy so concurrent requests don't mutate the master grid
    net = copy.deepcopy(MASTER_NET)

    # Power factor assumptions for reactive power
    PF_RES = 0.95   # Residential
    PF_COM = 0.92   # Commercial
    PF_IND = 0.85   # Industrial

    def kw_to_mw(kw): return round(kw / 1000.0, 6)
    def kw_to_mvar(kw, pf): return round((kw / 1000.0) * np.tan(np.arccos(pf)), 6)

    # Update loads (indices match creation order: 0=res, 1=com, 2=ind)
    net.load.at[0, "p_mw"]    = kw_to_mw(req.residential_kw)
    net.load.at[0, "q_mvar"]  = kw_to_mvar(req.residential_kw, PF_RES)

    net.load.at[1, "p_mw"]    = kw_to_mw(req.commercial_kw)
    net.load.at[1, "q_mvar"]  = kw_to_mvar(req.commercial_kw, PF_COM)

    net.load.at[2, "p_mw"]    = kw_to_mw(req.industrial_kw)
    net.load.at[2, "q_mvar"]  = kw_to_mvar(req.industrial_kw, PF_IND)

    # Update solar generation
    net.sgen.at[0, "p_mw"] = kw_to_mw(req.residential_solar_kw)
    net.sgen.at[1, "p_mw"] = kw_to_mw(req.commercial_solar_kw)
    net.sgen.at[2, "p_mw"] = kw_to_mw(req.industrial_solar_kw)

    # Run Newton-Raphson Power Flow
    try:
        pp.runpp(net, algorithm="nr", init="auto", max_iteration=50)
    except Exception as e:
        logger.error("Load flow failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Load flow diverged: {str(e)}")

    if not net.converged:
        raise HTTPException(status_code=500, detail="Load flow did not converge.")

    # ── EXTRACT RESULTS ──────────────────────────
    # Bus voltages
    bus_results = []
    for idx, row in net.res_bus.iterrows():
        bus_results.append({
            "index": int(idx),
            "name": net.bus.at[idx, "name"],
            "vm_pu": round(float(row["vm_pu"]), 4),
            "va_degree": round(float(row["va_degree"]), 2),
            "p_mw": round(float(row["p_mw"]), 6),
            "q_mvar": round(float(row["q_mvar"]), 6),
        })

    # Line results
    line_results = []
    for idx, row in net.res_line.iterrows():
        line_results.append({
            "index": int(idx),
            "name": net.line.at[idx, "name"],
            "loading_percent": round(float(row["loading_percent"]), 2),
            "p_from_mw": round(float(row["p_from_mw"]), 6),
            "p_to_mw": round(float(row["p_to_mw"]), 6),
            "pl_mw": round(float(row["pl_mw"]), 6),          # Active power loss
            "ql_mvar": round(float(row["ql_mvar"]), 6),       # Reactive power loss
            "i_ka": round(float(row["i_ka"]), 4),             # Current in kA
        })

    # Transformer results
    trafo_results = []
    for idx, row in net.res_trafo.iterrows():
        trafo_results.append({
            "index": int(idx),
            "name": net.trafo.at[idx, "name"],
            "loading_percent": round(float(row["loading_percent"]), 2),
            "pl_mw": round(float(row["pl_mw"]), 6),
            "ql_mvar": round(float(row["ql_mvar"]), 6),
            "i_hv_ka": round(float(row["i_hv_ka"]), 4),
            "i_lv_ka": round(float(row["i_lv_ka"]), 4),
        })

    # Aggregate metrics
    total_generation_kw = round(float(net.res_ext_grid["p_mw"].sum()) * 1000, 2)
    total_line_loss_kw  = round(float(net.res_line["pl_mw"].sum()) * 1000, 2)
    total_trafo_loss_kw = round(float(net.res_trafo["pl_mw"].sum()) * 1000, 2)
    total_loss_kw       = round(total_line_loss_kw + total_trafo_loss_kw, 2)
    total_demand_kw     = round(
        req.residential_kw + req.commercial_kw + req.industrial_kw, 2
    )
    total_solar_kw      = round(
        req.residential_solar_kw + req.commercial_solar_kw + req.industrial_solar_kw, 2
    )

    return {
        "converged": True,
        "summary": {
            "total_generation_kw": total_generation_kw,
            "total_demand_kw": total_demand_kw,
            "total_solar_kw": total_solar_kw,
            "total_line_loss_kw": total_line_loss_kw,
            "total_trafo_loss_kw": total_trafo_loss_kw,
            "total_loss_kw": total_loss_kw,
            "loss_percent": round((total_loss_kw / max(total_generation_kw, 0.01)) * 100, 2),
        },
        "buses": bus_results,
        "lines": line_results,
        "transformers": trafo_results,
    }


# ══════════════════════════════════════════════════
#  PHASE 4: FAULT / SHORT-CIRCUIT ANALYSIS
# ══════════════════════════════════════════════════

@app.post("/api/fault")
@app.post("/api/fault/")
def run_fault_analysis(req: FaultRequest):
    """
    Simulates a fault at the specified bus using IEC 60909.
    Returns fault currents (kA) and voltage drops across the network.
    """
    net = copy.deepcopy(MASTER_NET)

    # Validate bus index
    if req.bus_index < 0 or req.bus_index >= len(net.bus):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid bus_index: {req.bus_index}. Valid range: 0-{len(net.bus)-1}"
        )

    # Map fault types
    fault_map = {
        "3ph": "3ph",
        "2ph": "2ph",
        "1ph": "1ph",
    }
    fault = fault_map.get(req.fault_type)
    if not fault:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid fault_type: '{req.fault_type}'. Use '3ph', '2ph', or '1ph'."
        )

    try:
        # Run short-circuit calculation (IEC 60909)
        sc.calc_sc(net, bus=req.bus_index, fault=fault, ip=True, ith=True)
    except Exception as e:
        logger.error("Fault analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Short-circuit calculation failed: {str(e)}")

    # Extract fault results
    fault_results = []
    for idx, row in net.res_bus_sc.iterrows():
        fault_results.append({
            "bus_index": int(idx),
            "bus_name": net.bus.at[idx, "name"],
            "ikss_ka": round(float(row.get("ikss_ka", 0)), 4),   # Initial symmetrical SC current
            "ip_ka": round(float(row.get("ip_ka", 0)), 4),       # Peak SC current
            "ith_ka": round(float(row.get("ith_ka", 0)), 4),     # Thermal equivalent SC current
        })

    # Determine which buses are downstream of the fault
    # (simplified: buses with index >= fault bus in our radial topology)
    downstream_buses = [
        int(idx) for idx in net.bus.index if idx >= req.bus_index
    ]

    # Identify affected lines
    affected_lines = []
    for idx, row in net.line.iterrows():
        if int(row["from_bus"]) in downstream_buses or int(row["to_bus"]) in downstream_buses:
            affected_lines.append({
                "index": int(idx),
                "name": row["name"],
                "from_bus": int(row["from_bus"]),
                "to_bus": int(row["to_bus"]),
            })

    # Fault bus specific data
    faulted_bus = None
    for r in fault_results:
        if r["bus_index"] == req.bus_index:
            faulted_bus = r
            break

    return {
        "fault_type": req.fault_type,
        "faulted_bus_index": req.bus_index,
        "faulted_bus_name": net.bus.at[req.bus_index, "name"],
        "fault_current": faulted_bus,
        "all_bus_sc": fault_results,
        "downstream_buses": downstream_buses,
        "affected_lines": affected_lines,
        "breaker_trip": True,
        "blackout_zones": [net.bus.at[i, "name"] for i in downstream_buses if i != req.bus_index],
    }


# ══════════════════════════════════════════════════
#  ENTRYPOINT (for Railway)
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
