"""serving-stack: the FastAPI service (week 2, CPU, tiny model).

Includes full OpenAI-compatible non-streaming and streaming support.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    HealthResponse,
    ModelCard,
    ModelList,
    ResponseMessage,
    Usage,
)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
API_KEY = os.environ.get("API_KEY", "")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))

app = FastAPI(title="serving-stack", version="wk2")

# Load once at import time. CPU only this week.
print(f"loading {MODEL_ID} on cpu ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.to("cpu")
model.eval()
print("model ready")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness and readiness."""
    return HealthResponse(status="ok", model=MODEL_ID)


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------
@app.get("/v1/models", response_model=ModelList)
def list_models() -> ModelList:
    """List the served model id(s)."""
    now = int(time.time())
    card = ModelCard(
        id=MODEL_ID,
        created=now,
        owned_by="serving-stack"
    )
    return ModelList(data=[card])


# ---------------------------------------------------------------------------
# POST /v1/chat/completions (Non-Streaming + Streaming)
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
def chat_completions(
    req: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
):
    if API_KEY:
        expected = f"Bearer {API_KEY}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Invalid API key")
    # 1. Build the prompt with the chat template
    messages_dict = [m.model_dump() for m in req.messages]
    model_inputs = tokenizer.apply_chat_template(
        messages_dict,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True  # Returns a dict with both input_ids and attention_mask
    )
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    prompt_tokens = input_ids.shape[1]

    # --- STREAMING PATH ---
    if req.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,  # <-- ADD THIS LINE
            "streamer": streamer,
            "max_new_tokens": min(req.max_tokens if req.max_tokens is not None else 128, MAX_TOKENS),
            "pad_token_id": tokenizer.eos_token_id,
        }
        if req.temperature and req.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = req.temperature
        else:
            gen_kwargs["do_sample"] = False

        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        def stream_generator():
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created_time = int(time.time())
            
            # Initial chunk with assistant role
            initial_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(initial_chunk)}\n\n"

            for new_text in streamer:
                if new_text:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": req.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": new_text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

            # Final chunk
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # --- NON-STREAMING PATH ---
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": min(req.max_tokens if req.max_tokens is not None else 128, MAX_TOKENS),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if req.temperature and req.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = req.temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        out = model.generate(**gen_kwargs)

    new_tokens = out[0][prompt_tokens:]
    completion_tokens = len(new_tokens)
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    effective_max_tokens = min(req.max_tokens if req.max_tokens is not None else 128, MAX_TOKENS)
    finish_reason = "length" if completion_tokens >= effective_max_tokens else "stop"

    choice = Choice(
        index=0,
        message=ResponseMessage(role="assistant", content=text),
        finish_reason=finish_reason
    )
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[choice],
        usage=usage
    )