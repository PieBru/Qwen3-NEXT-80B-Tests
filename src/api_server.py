"""
FastAPI server with MoE-aware endpoints
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import time
import uuid
import json
import logging
from datetime import datetime

from src.inference import MoEInferencePipeline
from src.model_loader import ModelLoader
from src.config import default_config

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Qwen3-Local MoE API",
    description="API for MoE-aware Qwen3-80B inference",
    version="1.0.0"
)

# Add CORS middleware
# Configure CORS - adjust origins as needed for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:8080",  # Alternative local port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        # Add your production domains here when deploying
        # "https://yourdomain.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Request/Response models
class GenerateRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(
        ...,
        description="Input prompt",
        min_length=1,
        max_length=10000  # Reasonable limit for input
    )
    max_tokens: Optional[int] = Field(
        100,
        description="Maximum tokens to generate",
        ge=1,
        le=4096  # Maximum output tokens
    )
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.8, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(20, ge=1)
    stream: Optional[bool] = Field(False, description="Stream response")
    optimize_experts: Optional[bool] = Field(True, description="Optimize expert placement")


class BatchGenerateRequest(BaseModel):
    """Request model for batch generation"""
    prompts: List[str] = Field(
        ...,
        description="List of prompts",
        min_items=1,
        max_items=100  # Reasonable batch size limit
    )
    max_tokens: Optional[int] = Field(100, ge=1, le=4096)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.8, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(20)
    optimize_experts: Optional[bool] = Field(True)


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(
        ...,
        description="Message role (user/assistant/system)",
        pattern="^(user|assistant|system)$"  # Validate role values
    )
    content: str = Field(
        ...,
        description="Message content",
        min_length=1,
        max_length=10000  # Reasonable message length limit
    )


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = Field("qwen3-80b", description="Model name")
    messages: List[ChatMessage] = Field(
        ...,
        description="Chat messages",
        min_items=1,
        max_items=100  # Reasonable conversation length limit
    )
    max_tokens: Optional[int] = Field(100, ge=1, le=4096)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.8, ge=0.0, le=1.0)
    stream: Optional[bool] = Field(False)
    n: Optional[int] = Field(1, description="Number of completions", ge=1, le=10)


class ExpertOptimizationRequest(BaseModel):
    """Request model for expert optimization control"""
    enable: bool = Field(True, description="Enable expert optimization")
    cache_size: Optional[int] = Field(5, description="Number of experts to cache per layer")
    profiling_samples: Optional[int] = Field(100, description="Number of samples for profiling")


class GenerateResponse(BaseModel):
    """Response model for text generation"""
    text: str
    tokens_generated: Optional[int] = None
    generation_time: Optional[float] = None
    expert_cache_hits: Optional[int] = None


class BatchGenerateResponse(BaseModel):
    """Response model for batch generation"""
    results: List[str]
    total_tokens: int
    generation_time: float


class ModelService:
    """Service class for model operations"""

    def __init__(self, model=None, tokenizer=None, expert_manager=None):
        """Initialize model service"""
        self.model = model
        self.tokenizer = tokenizer
        self.expert_manager = expert_manager
        self.pipeline = None

        if model and tokenizer and expert_manager:
            self.pipeline = MoEInferencePipeline(
                model=model,
                tokenizer=tokenizer,
                expert_manager=expert_manager
            )
            logger.info("Model service initialized")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text"""
        if not self.pipeline:
            raise RuntimeError("Model not loaded")

        return self.pipeline.generate(prompt, **kwargs)

    def generate_stream(self, prompt: str, **kwargs):
        """Generate text with streaming"""
        if not self.pipeline:
            raise RuntimeError("Model not loaded")

        return self.pipeline.generate_stream(prompt, **kwargs)

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for batch"""
        if not self.pipeline:
            raise RuntimeError("Model not loaded")

        return self.pipeline.generate_batch(prompts, **kwargs)

    def chat_completion(
        self,
        messages: List[Dict],
        model: str = "qwen3",
        **kwargs
    ) -> Dict:
        """OpenAI-compatible chat completion"""
        if not self.pipeline:
            raise RuntimeError("Model not loaded")

        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)

        # Generate response
        response_text = self.pipeline.generate(prompt, **kwargs)

        # Format as OpenAI response
        return {
            "id": f"chat-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(self.tokenizer.encode(prompt)),
                "completion_tokens": len(self.tokenizer.encode(response_text)),
                "total_tokens": len(self.tokenizer.encode(prompt + response_text))
            }
        }

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert chat messages to prompt"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    def get_expert_stats(self) -> Dict:
        """Get expert cache statistics"""
        if not self.expert_manager:
            return {}

        return self.expert_manager.get_cache_stats()

    def get_model_info(self) -> Dict:
        """Get model information"""
        if not self.model:
            return {}

        config = self.model.config
        return {
            "model_name": "qwen3-80b-moe-bnb",
            "num_parameters": "80B",
            "activated_parameters": "3B",
            "num_layers": config.num_hidden_layers,
            "num_experts": getattr(config, 'num_experts', 64),
            "context_length": config.max_position_embeddings,
            "quantization": "4-bit BitsAndBytes",
            "device": str(next(self.model.parameters()).device)
        }

    def get_memory_stats(self) -> Dict:
        """Get memory statistics"""
        if not self.pipeline:
            return {}

        return self.pipeline.memory_manager.get_memory_stats()

    def optimize_experts(self, enable: bool, cache_size: int = 5):
        """Control expert optimization"""
        if not self.expert_manager:
            return

        if enable:
            self.expert_manager.num_cached_experts_per_layer = cache_size
            logger.info(f"Expert optimization enabled with cache size {cache_size}")
        else:
            self.expert_manager.clear_cache()
            logger.info("Expert optimization disabled")


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""

    try:
        # Load model
        loader = ModelLoader(default_config)
        model, tokenizer = loader.load_model()

        # Initialize service
        logger.info(f"Initializing ModelService with model={model is not None}, tokenizer={tokenizer is not None}, expert_manager={hasattr(loader, 'expert_cache_manager')}")
        app.state.model_service = ModelService(
            model=model,
            tokenizer=tokenizer,
            expert_manager=loader.expert_cache_manager if hasattr(loader, 'expert_cache_manager') else None
        )
        logger.info(f"ModelService initialized. Pipeline created: {app.state.model_service.pipeline is not None}")
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Cannot start server without a working model. Exiting...")
        # Exit with error code - server should not start if model fails to load
        import sys
        sys.exit(1)


@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "service": "Qwen3-Local MoE API",
        "version": "1.0.0",
        "model": "qwen3-80b-moe-bnb",
        "status": "ready" if hasattr(app.state, 'model_service') and app.state.model_service and app.state.model_service.pipeline else "not loaded"
    }


@app.get("/api/v1/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": hasattr(app.state, 'model_service') and app.state.model_service is not None and app.state.model_service.pipeline is not None
    }


@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text endpoint"""
    try:
        if not hasattr(app.state, 'model_service') or not app.state.model_service or not app.state.model_service.pipeline:
            raise HTTPException(status_code=503, detail="Model not loaded")

        start_time = time.time()

        # Generate text
        if request.stream:
            # Return streaming response
            def stream_generator():
                for token in app.state.model_service.generate_stream(
                    request.prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k
                ):
                    yield json.dumps({"token": token}) + "\n"

            return StreamingResponse(
                stream_generator(),
                media_type="application/x-ndjson"
            )
        else:
            # Regular generation
            text = app.state.model_service.generate(
                request.prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                optimize_experts=request.optimize_experts
            )

            generation_time = time.time() - start_time

            # Get expert stats
            expert_stats = app.state.model_service.get_expert_stats()

            return GenerateResponse(
                text=text,
                tokens_generated=len(app.state.model_service.tokenizer.encode(text)),
                generation_time=generation_time,
                expert_cache_hits=expert_stats.get('cached_experts', 0)
            )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate/batch", response_model=BatchGenerateResponse)
async def generate_batch(request: BatchGenerateRequest):
    """Batch generation endpoint"""
    try:
        if not hasattr(app.state, 'model_service') or not app.state.model_service or not app.state.model_service.pipeline:
            raise HTTPException(status_code=503, detail="Model not loaded")

        start_time = time.time()

        # Generate batch
        results = app.state.model_service.generate_batch(
            request.prompts,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            optimize_experts=request.optimize_experts
        )

        generation_time = time.time() - start_time

        # Calculate total tokens
        total_tokens = sum(
            len(app.state.model_service.tokenizer.encode(text))
            for text in results
        )

        return BatchGenerateResponse(
            results=results,
            total_tokens=total_tokens,
            generation_time=generation_time
        )

    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint"""
    try:
        if not hasattr(app.state, 'model_service') or not app.state.model_service or not app.state.model_service.pipeline:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert messages to dict format
        messages = [msg.dict() for msg in request.messages]

        # Generate response
        response = app.state.model_service.chat_completion(
            messages=messages,
            model=request.model,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream
        )

        return response

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/expert-stats")
async def get_expert_stats():
    """Get expert cache statistics"""
    try:
        if not hasattr(app.state, 'model_service') or not app.state.model_service:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return app.state.model_service.get_expert_stats()

    except Exception as e:
        logger.error(f"Expert stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/expert-optimization")
async def configure_expert_optimization(request: ExpertOptimizationRequest):
    """Configure expert optimization"""
    try:
        if not hasattr(app.state, 'model_service') or not app.state.model_service:
            raise HTTPException(status_code=503, detail="Model not loaded")

        app.state.model_service.optimize_experts(
            enable=request.enable,
            cache_size=request.cache_size
        )

        return {
            "status": "configured",
            "optimization_enabled": request.enable,
            "cache_size": request.cache_size
        }

    except Exception as e:
        logger.error(f"Expert optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/model")
async def get_model_info():
    """Get model information"""
    try:
        if not hasattr(app.state, 'model_service') or not app.state.model_service:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return app.state.model_service.get_model_info()

    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/memory")
async def get_memory_stats():
    """Get memory statistics"""
    try:
        if not hasattr(app.state, 'model_service') or not app.state.model_service:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return app.state.model_service.get_memory_stats()

    except Exception as e:
        logger.error(f"Memory stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming"""
    await websocket.accept()
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            if data.get("type") == "generate":
                # Stream generation
                prompt = data.get("prompt", "")
                max_tokens = data.get("max_tokens", 100)

                for token in app.state.model_service.generate_stream(
                    prompt,
                    max_new_tokens=max_tokens
                ):
                    await websocket.send_json({
                        "type": "token",
                        "content": token
                    })

                await websocket.send_json({
                    "type": "complete"
                })

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)