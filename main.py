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
    version="1.1.0",
    description="SC-Hardened pandapower API.",
)

# Global CORS — Allow all origins to support Vercel/DigitalZen/Local
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

    # 1. BUS PRESET (HV -> MV1 -> MV2 -> Dist)
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
        x0x_max=0.1, r0x0_max=0.1, # Zero sequence resistance/reactance ratio
    )

    # 3. TRANSFORMERS (With Z-sequence impedance for faults)
    # T1: 110/33kV
    pp.create_transformer_from_parameters(
        net, hv_bus=bus_hv, lv_bus=bus_mv1,
        sn_mva=63, vn_hv_kv=110, vn_lv_kv=33,
        vkr_percent=0.1, vk_percent=10, pfe_kw=20, i0_percent=0.1,
        vk0_percent=10, vkr0_percent=0.1,
        name="T1"
    )
    # T2: 33/11kV 
    pp.create_transformer_from_parameters(
        net, hv_bus=bus_mv1, lv_bus=bus_mv2,
        sn_mva=25, vn_hv_kv=33, vn_lv_kv=11,
        vkr_percent=0.1, vk_percent=10, pfe_kw=10, i0_percent=0.1,
        vk0_percent=10, vkr0_percent=0.1,
        name="T2"
    )

    # CRITICAL FIX: To avoid Pandas LossySetitemError, force column to object before setting string
    if "vector_group" in net.trafo.columns:
        net.trafo["vector_group"] = net.trafo["vector_group"].astype(object)
        net.trafo.at[0, "vector_group"] = "Dyn"
        net.trafo.at[1, "vector_group"] = "Dyn"
    else:
        # Fallback if column doesn't exist yet
        net.trafo["vector_group"] = ["Dyn", "Dyn"]


    # 4. LINES (Manually enriched with sequence parameters)
    # Standard cable metrics for 11kV: NAYY 4x50 SE (R1=0.642, X1=0.083 per km)
    r1 = 0.642; x1 = 0.083; c1 = 210
    # Engineering heuristic: Z0 ≈ 4 * Z1
    r0 = r1 * 4; x0 = x1 * 4; c0 = c1

    pp.create_line_from_parameters(
        net, from_bus=bus_mv2, to_bus=bus_res, length_km=5.0,
        r_ohm_per_km=r1, x_ohm_per_km=x1, c_nf_per_km=c1, max_i_ka=0.21,
        r0_ohm_per_km=r0, x0_ohm_per_km=x0, c0_nf_per_km=c0, # Mandatory for SC
        name="Feeder → residential"
    )
    pp.create_line_from_parameters(
        net, from_bus=bus_mv2, to_bus=bus_com, length_km=4.0,
        r_ohm_per_km=r1, x_ohm_per_km=x1, c_nf_per_km=c1, max_i_ka=0.21,
        r0_ohm_per_km=r0, x0_ohm_per_km=x0, c0_nf_per_km=c0, # Mandatory for SC
        name="Feeder → commercial"
    )
    pp.create_line_from_parameters(
        net, from_bus=bus_mv2, to_bus=bus_ind, length_km=8.0,
        r_ohm_per_km=r1, x_ohm_per_km=x1, c_nf_per_km=c1, max_i_ka=0.21,
        r0_ohm_per_km=r0, x0_ohm_per_km=x0, c0_nf_per_km=c0, # Mandatory for SC
        name="Feeder → industrial"
    )

    # 5. LOADS
    pp.create_load(net, bus=bus_res, p_mw=0.096, q_mvar=0.031, name="Residential Load")
    pp.create_load(net, bus=bus_com, p_mw=0.150, q_mvar=0.064, name="Commercial Load")
    pp.create_load(net, bus=bus_ind, p_mw=0.950, q_mvar=0.588, name="Industrial Load")

    # 6. SOLAR (With sn_mva for fault contribution)
    pp.create_sgen(net, bus=bus_res, p_mw=0.02, sn_mva=0.025, name="Res Solar")
    pp.create_sgen(net, bus=bus_com, p_mw=0.02, sn_mva=0.025, name="Com Solar")
    pp.create_sgen(net, bus=bus_ind, p_mw=0.05, sn_mva=0.060, name="Ind Solar")

    return net

MASTER_NET = create_city_grid()

# ══════════════════════════════════════════════════
#  API MODELS
# ══════════════════════════════════════════════════
class LoadFlowRequest(BaseModel):
    residential_kw: float
    commercial_kw: float
    industrial_kw: float
    residential_solar_kw: float
    commercial_solar_kw: float
    industrial_solar_kw: float

class FaultRequest(BaseModel):
    bus_index: int
    fault_type: str

# ══════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "ok", "version": "1.1.0"}

@app.post("/api/loadflow")
@app.post("/api/loadflow/")
def run_load_flow(req: LoadFlowRequest):
    net = copy.deepcopy(MASTER_NET)
    
    def kw_to_mw(kw): return round(kw / 1000.0, 6)
    
    # Update loads
    net.load.at[0, "p_mw"] = kw_to_mw(req.residential_kw)
    net.load.at[1, "p_mw"] = kw_to_mw(req.commercial_kw)
    net.load.at[2, "p_mw"] = kw_to_mw(req.industrial_kw)
    
    # Update sgen
    net.sgen.at[0, "p_mw"] = kw_to_mw(req.residential_solar_kw)
    net.sgen.at[1, "p_mw"] = kw_to_mw(req.commercial_solar_kw)
    net.sgen.at[2, "p_mw"] = kw_to_mw(req.industrial_solar_kw)

    try:
        pp.runpp(net)
        
        lines = []
        for idx, row in net.res_line.iterrows():
            lines.append({
                "index": int(idx),
                "loading_percent": round(float(row["loading_percent"]), 2),
                "p_from_mw": round(float(row["p_from_mw"]), 6),
                "pl_mw": round(float(row["pl_mw"]), 6),
                "i_ka": round(float(row["i_ka"]), 4)
            })

        return {
            "converged": True,
            "summary": {
                "total_generation_kw": round(float(net.res_ext_grid["p_mw"].sum()) * 1000, 2),
                "total_demand_kw": round((req.residential_kw + req.commercial_kw + req.industrial_kw), 2),
                "total_loss_kw": round(float(net.res_line["pl_mw"].sum() + net.res_trafo["pl_mw"].sum()) * 1000, 2),
                "loss_percent": round((net.res_line["pl_mw"].sum() / max(net.res_ext_grid["p_mw"].sum(), 0.01)) * 100, 2),
            },
            "lines": lines
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/fault")
@app.post("/api/fault/")
def run_fault_analysis(req: FaultRequest):
    net = copy.deepcopy(MASTER_NET)
    
    # Check bus bound
    if req.bus_index < 0 or req.bus_index >= len(net.bus):
        raise HTTPException(status_code=400, detail="Invalid bus index")

    try:
        # IEC 60909 SC Calculation
        sc.calc_sc(net, bus=req.bus_index, fault=req.fault_type, ip=True, ith=True)
        
        # SC Results
        res = net.res_bus_sc.loc[req.bus_index]
        
        # Identify blackout zones (simplified: all load buses downstream of fault)
        # In our radial grid: 2(sub) -> 3(res), 4(com), 5(ind)
        downstream = [3, 4, 5] if req.bus_index <= 2 else [req.bus_index]
        
        return {
            "faulted_bus_index": req.bus_index,
            "faulted_bus_name": net.bus.at[req.bus_index, "name"],
            "fault_current": {
                "ikss_ka": round(float(res["ikss_ka"]), 4),
                "ip_ka": round(float(res.get("ip_ka", 0)), 4)
            },
            "breaker_trip": True,
            "downstream_buses_offline": downstream,
            "blackout_zones": [net.bus.at[i, "name"] for i in downstream],
            "faulted_line_index": 0 # Default line trip for visual
        }
    except Exception as e:
        logger.error("FAULT ERROR: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Short-circuit calculation failed: {str(e)}")

# ENTRYPOINT
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
