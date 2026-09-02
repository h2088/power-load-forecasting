# Power Forecast Serving And Deployment Plan

## 1. Why this structure

This implementation keeps the same basic teaching pattern as `simple_web_service-main`, but maps it to the power forecasting use case:

| Reference project | This project | Responsibility |
|---|---|---|
| `backend/app.py` | `backend/app.py` | HTTP routes only |
| `backend/chat_logic.py` | `backend/forecast_logic.py` | Business logic: train scheme 1 and generate a 96-slot forecast |
| `backend/storage.py` | `backend/data_store.py` | Data loading and dataset metadata |
| `frontend/frontend.py` | `frontend/frontend.py` | Streamlit dashboard |
| `docker-compose.yml` | `docker-compose.yml` | Local two-container orchestration |

The result is still easy to understand, but more suitable for an ML service:

- The backend knows the D-6 constraint and the 28-day rolling training window.
- The model serving layer is isolated from the route layer.
- The runtime thread cap is fixed to avoid `HistGradientBoostingRegressor` issues in constrained environments.

## 2. Chosen serving architecture

### Local runnable architecture

```text
Browser
  -> Streamlit frontend
  -> Flask backend
  -> scheme 1 forecasting logic
  -> total_consumption.csv
```

### Production architecture

```text
Public ALB
  -> Frontend service (Streamlit container)
  -> Backend service (Flask container)
  -> CSV data / forecast artifacts in S3
  -> CloudWatch logs
  -> EventBridge scheduled batch forecast task
```

## 3. Why this is scientifically reasonable for scheme 1

This deployment is aligned with the modeling assumptions already accepted in the project:

- Forecast target is always `D`.
- Latest allowed actual data is always `D-6`.
- Training window is the latest 28 calendar days ending at `D-6`.
- Inference uses scheme 1 unchanged:
  - regime-aware features
  - two-stage level + shape prediction
  - anomaly-aware lag handling

So the deployment layer does not distort the modeling logic. It only wraps the existing model in a stable service interface.

## 4. Runnable local commands

### Backend

```powershell
python -m backend.app
```

### Frontend

```powershell
streamlit run frontend/frontend.py
```

### Docker compose

```powershell
docker compose up --build
```

Then open:

- Backend health: `http://localhost:5000/health`
- Frontend dashboard: `http://localhost:8501`

## 5. API contract

### `GET /health`

Returns backend health status.

### `GET /api/model-info`

Returns:

- model name
- D-6 and window settings
- feature groups
- regime segments
- dataset metadata

### `POST /api/predict`

Request:

```json
{
  "target_date": "2026-04-05"
}
```

Response:

- target date
- cutoff date
- training window
- 96-slot predictions
- actual values when available
- quality warnings

## 6. Production recommendation on AWS

### Recommended path

Use **ECS Fargate** for both containers.

Why this is the most balanced choice:

- More production-ready than running ad hoc EC2 scripts
- Easier long-term than keeping everything inside a notebook
- Better than pure App Runner when you also want scheduled batch jobs and tighter control over networking

### Concrete AWS layout

1. Push backend and frontend images to ECR.
2. Run two ECS Fargate services:
   - `power-forecast-backend`
   - `power-forecast-frontend`
3. Put an ALB in front:
   - `/` routes to frontend
   - `/api/*` routes to backend
4. Store `total_consumption.csv` and daily forecast artifacts in S3.
5. Use EventBridge to trigger a daily ECS task:
   - run `python -m backend.batch_forecast`
   - output forecast csv/json into S3
6. Use CloudWatch for logs and alarms.

The public repository intentionally does not contain source load data. For the
local Compose path, place `total_consumption.csv` under `data/`; the backend
container receives that directory as a read-only volume. In AWS, mount or
download the corresponding private S3 object into the task instead of baking
it into the image.

## 7. Online prediction vs scheduled batch

For an internal tool, the current backend can train on demand and return the forecast directly.

For production, the recommended operating mode is:

1. Data lands in S3.
2. Scheduled forecast job runs once per day.
3. Backend API serves the latest stored forecast first.
4. Manual on-demand retraining remains as an admin capability.

This is better because it reduces request latency, makes forecasts reproducible, and avoids training the model repeatedly for the same target date.

## 8. What is already implemented in this repo

- Flask backend service
- Streamlit frontend dashboard
- Dockerfiles for both services
- `docker-compose.yml`
- Batch forecast CLI for scheduled jobs

That means the project is already at a solid MVP-to-production transition point: it runs locally now, and the next deployment step is mainly AWS infrastructure, not application redesign.
