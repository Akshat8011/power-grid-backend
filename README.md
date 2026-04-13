# Power Grid Digital Twin — Backend Engine

**pandapower** + **FastAPI** physics engine for the 3D React Three Fiber digital twin.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/`  | Health check — returns grid metadata |
| `POST` | `/api/loadflow` | Newton-Raphson load flow analysis |
| `POST` | `/api/fault` | IEC 60909 short-circuit fault simulation |

## Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Railway Deployment

1. Push this folder as a separate GitHub repository.
2. Connect it to Railway.
3. Set the start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add env var `FRONTEND_URL` = your Vercel production URL.

## API Examples

### Load Flow
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

### Fault Analysis
```json
POST /api/fault
{
  "bus_index": 2,
  "fault_type": "3ph"
}
```
