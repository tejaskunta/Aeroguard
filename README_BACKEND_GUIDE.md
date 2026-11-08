# AeroShield Backend Guide (Additional Readme)

This file is an additional project readme and does not replace the main README.

## Overview

AeroShield is an AI-assisted drone landmine detection and mapping project.
Current implementation in this repository focuses on a clean FastAPI backend foundation.

> **Stale snapshot — see the live repository.** The `backend/` and `jetson/`
> directories here are a Week-1/2 snapshot merged in at commit `9c1d287`
> ("Integrate AeroShield backend architecture with SafeMine YOLOv8 training
> pipeline"). AeroShield is now its own repository, `tejaskunta/AeroShield`,
> which has moved well past this point (full API-key auth, PostgreSQL/PostGIS
> models, RAG copilot, a React/TypeScript mission console, and an 11-module
> TensorRT edge pipeline). Everything below describes **only** the snapshot in
> this repo. Do not cite numbers from this snapshot as current AeroShield
> behaviour.
>
> Note also that this repository is **AeroGuard**, the landmine-detection
> research paper. AeroShield is the companion deployment/system project; the two
> names are not interchangeable.

## Current Backend Progress

- Phase 1 complete: FastAPI app, root endpoint, health endpoint, Swagger docs.
- Phase 2 complete: image upload detection endpoint with a dummy/mock detector.

## Backend Folder

```text
backend/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── detection.py
│   ├── services/
│   │   └── detection_service.py
│   └── schemas/
│       └── detection.py
└── requirements.txt
```

## Run (Windows PowerShell)

From project root:

```powershell
& ".\\.venv\\Scripts\\python.exe" -m pip install -r .\\backend\\requirements.txt
& ".\\.venv\\Scripts\\python.exe" -m uvicorn app.main:app --reload --app-dir .\\backend
```

From backend folder:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

## Endpoints

- GET / -> {"message": "AeroShield API is running"}
- GET /health -> {"status": "healthy"}
- POST /api/detect -> mock detection response for uploaded image

## Swagger

Open:

- http://127.0.0.1:8000/docs

## Important Note

The detection route currently uses a mock service.
Later, it can be swapped with a YOLO model service without breaking the API route contract.
