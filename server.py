from vllm import LLMEngine, SamplingParams, LLM

# from vllm.entrypoints.api_server import AsyncLLMEngine as AsyncLLMServer
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import uvicorn
import asyncio

# Initialize FastAPI app
app = FastAPI(title="Qwen 7B API Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS = 2048
TEMPERATURE = 0.7

# Initialize LLM Engine
engine = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=4,  # Adjust based on your GPU setup
    # dtype="float16",  # Use float16 for efficiency
    # trust_remote_code=True,  # Required for Qwen models
    max_model_len=MAX_TOKENS,
    # quantization="awq",
)

# Create sampling parameters
default_params = SamplingParams(
    temperature=TEMPERATURE, max_tokens=MAX_TOKENS, stop=["</s>", "<|endoftext|>"]
)


@app.post("/generate")
async def generate(request: Request) -> Dict[str, Any]:
    """
    Generate text based on the provided prompt.

    Expected JSON body:
    {
        "prompt": "Your prompt here",
        "max_tokens": 2048,  # optional
        "temperature": 0.7   # optional
    }
    """
    try:
        # Properly await the JSON data
        json_data = await request.json()
        prompt = json_data.get("prompt")
        if not prompt:
            return {"error": "No prompt provided"}

        # Override default parameters if provided
        sampling_params = SamplingParams(
            temperature=json_data.get("temperature", TEMPERATURE),
            max_tokens=json_data.get("max_tokens", MAX_TOKENS),
            stop=["</s>", "<|endoftext|>"],
        )

        try:
            # Generate response
            outputs = engine.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text

            return {
                "generated_text": generated_text,
                "usage": {
                    "prompt_tokens": len(
                        prompt.split()
                    ),  # Note: This is an approximation
                    "completion_tokens": len(
                        generated_text.split()
                    ),  # Note: This is an approximation
                    "total_tokens": len(prompt.split()) + len(generated_text.split()),
                },
            }
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}

    except Exception as e:
        return {"error": f"Invalid request: {str(e)}"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio")
