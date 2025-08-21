import os, time, json, io, requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Form, Query
from fastapi.responses import StreamingResponse

# ========= Config via ENV =========
WORKFLOW_JSON   = os.getenv("WORKFLOW_JSON", "JSON.json")  # keep this file next to app.py
INPUT_NODE_ID   = str(os.getenv("INPUT_NODE_ID", "54"))    # your "Load Image" node id
OUTPUT_NODE_ID  = str(os.getenv("OUTPUT_NODE_ID", "455"))  # node that exposes the mask
POLL_INTERVAL_S = float(os.getenv("POLL_INTERVAL_S", "5"))
POLL_TIMEOUT_S  = float(os.getenv("POLL_TIMEOUT_S", "120"))
MAX_RETRIES     = int(os.getenv("MAX_RETRIES", "3"))
API_KEY         = os.getenv("API_KEY", "mysecret123")    

app = FastAPI(title="Comfy Masking API", version="1.0.0")

def _require_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def _pick_comfy_url(form_url: str | None = None, header_url: str | None = None, query_url: str | None = None) -> str:
    """Pick the first non-empty COMFY_URL from form field > header > query param"""
    for url in [form_url, header_url, query_url]:
        if url and url.strip():
            # Normalize with trailing slash
            normalized = url.rstrip('/') + '/'
            return normalized
    # No URL provided - raise error since hardcoded fallback is unreliable
    raise HTTPException(status_code=400, detail="comfy_url must be provided via form field, header (x-comfy-url), or query parameter")

def _upload_to_comfy(img_bytes: bytes, original_name: str, base_url: str) -> str:
    stamped = f"{int(time.time()*1000)}_{original_name}"
    last_err = None
    for _ in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{base_url.rstrip('/')}/upload/image",
                files={"image": (stamped, img_bytes)},
                timeout=60,
            )
            r.raise_for_status()
            return stamped
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise HTTPException(status_code=502, detail=f"Upload failed: {last_err}")

def _queue_prompt(graph: dict, base_url: str) -> str:
    r = requests.post(f"{base_url.rstrip('/')}/prompt", json={"prompt": graph}, timeout=60)
    r.raise_for_status()
    pid = r.json().get("prompt_id")
    if not pid:
        raise HTTPException(status_code=500, detail="ComfyUI did not return prompt_id")
    return pid

def _poll_history(prompt_id: str, base_url: str) -> dict:
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        r = requests.get(f"{base_url.rstrip('/')}/history/{prompt_id}", timeout=60)
        if r.status_code == 200:
            hist = r.json()
            if hist and prompt_id in hist:
                outputs = hist[prompt_id].get("outputs", {})
                node = outputs.get(OUTPUT_NODE_ID)
                if not node:
                    for _, v in outputs.items():
                        if v.get("images"):
                            node = v
                            break
                if node and node.get("images"):
                    return hist
        time.sleep(POLL_INTERVAL_S)
    raise HTTPException(status_code=504, detail="Timed out waiting for ComfyUI")

def _fetch_first_image(history: dict, prompt_id: str, base_url: str) -> bytes:
    outputs = history[prompt_id]["outputs"]
    node = outputs.get(OUTPUT_NODE_ID)
    if not node:
        for _, v in outputs.items():
            if v.get("images"):
                node = v
                break
    if not node or not node.get("images"):
        raise HTTPException(status_code=500, detail="No output images found")

    info = node["images"][0]  # {'filename','subfolder','type'}
    params = {
        "filename": info["filename"],
        "subfolder": info.get("subfolder", ""),
        "type": info.get("type", "output"),
    }
    r = requests.get(f"{base_url.rstrip('/')}/view", params=params, timeout=120)
    r.raise_for_status()
    return r.content

@app.get("/health")
def health(
    x_api_key: str | None = Header(default=None),
    x_comfy_url: str | None = Header(default=None),
    comfy_url: str | None = Query(default=None)
):
    _require_key(x_api_key)
    
    effective_comfy = _pick_comfy_url(
        form_url=None,
        header_url=x_comfy_url,
        query_url=comfy_url
    )
    
    return {
        "ok": True,
        "comfy_url": effective_comfy,
        "workflow_json": WORKFLOW_JSON,
        "input_node_id": INPUT_NODE_ID,
        "output_node_id": OUTPUT_NODE_ID,
    }

@app.post("/mask")
async def mask(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
    x_comfy_url: str | None = Header(default=None),
    comfy_url: str | None = Form(default=None),
    comfy_url_query: str | None = Query(default=None, alias="comfy_url")
):
    _require_key(x_api_key)

    # Pick the effective COMFY_URL based on priority
    effective_comfy = _pick_comfy_url(
        form_url=comfy_url,
        header_url=x_comfy_url,
        query_url=comfy_url_query
    )

    # 0) read input
    img_bytes = await file.read()
    original_name = file.filename or "image.png"

    # 1) upload to ComfyUI
    comfy_name = _upload_to_comfy(img_bytes, original_name, effective_comfy)

    # 2) load workflow JSON & set the input node's image
    try:
        with open(WORKFLOW_JSON, "r", encoding="utf-8") as f:
            graph = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read workflow JSON '{WORKFLOW_JSON}': {e}")

    node = graph.get(INPUT_NODE_ID) or graph.get(int(INPUT_NODE_ID))
    if not node:
        raise HTTPException(status_code=500, detail=f"Input node {INPUT_NODE_ID} not found in workflow JSON")
    node.setdefault("inputs", {})["image"] = comfy_name

    # 3) queue + 4) poll
    pid = _queue_prompt(graph, effective_comfy)
    history = _poll_history(pid, effective_comfy)

    # 5) fetch image & stream back (filename == original)
    out_bytes = _fetch_first_image(history, pid, effective_comfy)
    return StreamingResponse(
        io.BytesIO(out_bytes),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{original_name}"'}
    )
