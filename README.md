# Power Grid Digital Twin — Backend Engine

**pandapower** + **FastAPI** physics engine for the 3D React Three Fiber digital twin.

**Live API:** [https://hehehe897-power-grid-backend.hf.space](https://hehehe897-power-grid-backend.hf.space)  
**Frontend:** [https://load-analysis.vercel.app](https://load-analysis.vercel.app) · [Load-Analysis-Project-](https://github.com/Akshat8011/Load-Analysis-Project-)

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Health check — grid metadata |
| `POST` | `/api/loadflow` | Newton–Raphson load-flow analysis |
| `POST` | `/api/fault` | IEC 60909 short-circuit fault simulation |

## Local development

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Set `FRONTEND_URL` to your UI origin (e.g. `https://load-analysis.vercel.app`) for CORS in production.

## Example — load flow

```json
POST /api/loadflow
{
  "residential_kw": 120,
  "commercial_kw": 200,
  "industrial_kw": 800,
  "residential_solar_kw": 15,
  "commercial_solar_kw": 20,
  "industrial_solar_kw": 40
}
```

## Example — fault study

```json
POST /api/fault
{
  "bus_index": 2,
  "fault_type": "3ph"
}
```

## Author

**Akshat Choudhary** — Electrical Engineering + Software  
[github.com/Akshat8011](https://github.com/Akshat8011)