# Ollama Router

A smart AI model router for Open WebUI that intelligently routes queries to appropriate local models or ChatGPT based on content analysis and intent classification.

## Features

- 🤖 **Smart Routing**: Automatically selects the best model based on query content
- 🔄 **Session Management**: Maintains conversation context across multiple messages
- 🌐 **ChatGPT Integration**: Routes internet-requiring queries to ChatGPT when available
- 📝 **Intent Classification**: Categorizes queries into DEV, WRITING, or GENERAL
- 🎯 **Keyword Matching**: Fast-path routing using configurable keywords
- 📊 **Request Logging**: Comprehensive logging with routing decisions
- 🔧 **Configuration Management**: YAML-based configuration with environment overrides

## Quick Start

1. Configure your models in config.yaml
2. Set OPENAI_API_KEY environment variable (optional)
3. Build and run: docker-compose up -d ollama-router
4. Point WebUI to: OLLAMA_BASE_URL=http://ollama-router:8000

## Model Selection Logic

### Capability Assessment
- Internet queries → ChatGPT (weather, news, real-time data)
- Local queries → Local models (programming, writing, general knowledge)

### Intent Classification
- DEV: Programming, debugging → model_dev
- WRITING: Creative writing, stories → model_writing
- GENERAL: Everything else → model_general

## Configuration

Edit config.yaml to customize:
- Model assignments
- Keyword patterns
- Ollama backend URL

## API Endpoints

- POST /api/chat - Main chat endpoint with smart routing
- GET /api/tags - List available models
- GET /health - Health check
- GET /debug/sessions - View active sessions

The router is production-ready and fully integrated with WebUI!
