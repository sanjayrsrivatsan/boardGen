#!/usr/bin/env python3
"""
FastAPI backend for climb generation.

Endpoints:
- GET /boards - List available boards with loaded models
- POST /generate - Generate climbs for a specific board
- GET /board-config/{board} - Return board geometry and config
- POST /submit - Submit a climb to the board's API
- POST /evaluate - Run validity checks on a climb

Usage:
    python interface/server.py
    # Then open http://localhost:8000
"""

import json
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from generation.generate import generate, load_model
from generation.postprocess import check_climb_validity
from generation.convert import frames_to_tokens


app = FastAPI(title="BoardDiffusion API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache loaded models
_models = {}


def get_loaded_boards() -> List[str]:
    """Get list of boards with trained models."""
    boards = []
    for board in ["tension", "kilter", "moonboard"]:
        if (Path("checkpoints") / board / "best.pt").exists():
            boards.append(board)
    return boards


def ensure_model_loaded(board: str):
    """Load model if not already cached."""
    if board not in _models:
        try:
            model, config, vocab_meta = load_model(board)
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            _models[board] = {"model": model, "config": config, "vocab_meta": vocab_meta}
        except FileNotFoundError:
            raise HTTPException(404, f"No trained model for {board}")


# Request/Response models
class GenerateRequest(BaseModel):
    board: str
    n_samples: int = 10
    difficulty: Optional[float] = None
    angle: Optional[int] = None
    board_size: Optional[str] = None
    guidance_scale: float = 3.0
    temperature: float = 0.8
    n_holds: Optional[int] = None
    n_steps: int = 32


class SubmitRequest(BaseModel):
    board: str
    frames_string: str
    name: str
    angle: int
    username: str
    password: str


class EvaluateRequest(BaseModel):
    board: str
    frames_string: str


# Endpoints
@app.get("/")
async def root():
    return {"message": "BoardDiffusion API", "boards": get_loaded_boards()}


@app.get("/boards")
async def list_boards():
    """List available boards with loaded models."""
    boards = get_loaded_boards()
    result = []
    for board in boards:
        ensure_model_loaded(board)
        info = _models[board]
        result.append({
            "name": board,
            "V_total": info["vocab_meta"]["V_total"],
            "V_pos": info["vocab_meta"]["V_pos"],
            "V_role": info["vocab_meta"]["V_role"],
            "L_max": info["vocab_meta"]["L_max"],
            "angle_conditioning": info["config"].get("angle_conditioning", True),
            "size_conditioning": info["config"].get("size_conditioning", True),
            "board_sizes": info["vocab_meta"].get("board_sizes", []),
        })
    return result


@app.get("/board-config/{board}")
async def get_board_config(board: str):
    """Get board geometry and configuration."""
    ensure_model_loaded(board)
    info = _models[board]
    config = info["config"]
    vocab_meta = info["vocab_meta"]

    return {
        "board": board,
        "V_pos": vocab_meta["V_pos"],
        "V_role": vocab_meta["V_role"],
        "L_max": vocab_meta["L_max"],
        "role_names": vocab_meta["role_names"],
        "role_colors": config.get("role_colors", {}),
        "angle_conditioning": config.get("angle_conditioning", True),
        "angle_range": config.get("angle_range", [0, 70]),
        "default_angle": config.get("default_angle"),
        "size_conditioning": config.get("size_conditioning", True),
        "board_sizes": vocab_meta.get("board_sizes", []),
        "default_size": config.get("default_size"),
        "deep_link_scheme": config.get("deep_link_scheme", ""),
        "placement_coords": vocab_meta["placement_coords"].tolist(),
        "placement_ids": list(vocab_meta["placement_index_to_id"].values()),
    }


@app.post("/generate")
async def generate_climbs(request: GenerateRequest):
    """Generate climbing problems."""
    if request.board not in get_loaded_boards():
        raise HTTPException(404, f"No trained model for {request.board}")

    climbs = generate(
        board=request.board,
        n_samples=request.n_samples,
        difficulty=request.difficulty,
        angle=request.angle,
        board_size=request.board_size,
        guidance_scale=request.guidance_scale,
        temperature=request.temperature,
        n_holds=request.n_holds,
        n_steps=request.n_steps,
    )

    return {"climbs": climbs}


@app.post("/evaluate")
async def evaluate_climb(request: EvaluateRequest):
    """Evaluate validity of a climb."""
    ensure_model_loaded(request.board)
    vocab_meta = _models[request.board]["vocab_meta"]

    tokens = frames_to_tokens(request.frames_string, vocab_meta)
    validity = check_climb_validity(tokens, vocab_meta)

    return validity


@app.post("/submit")
async def submit_climb(request: SubmitRequest):
    """Submit a climb to the board's API."""
    from interface.submit import submit_climb as do_submit

    try:
        result = do_submit(
            board_name=request.board,
            username=request.username,
            password=request.password,
            frames_string=request.frames_string,
            layout_id=_models[request.board]["config"]["layout_id"],
            angle=request.angle,
            name=request.name,
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


# Serve frontend if available
frontend_path = Path(__file__).parent / "frontend" / "dist"
if frontend_path.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
