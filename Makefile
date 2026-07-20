SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE ?= docker compose
PROFILE ?= cuda
REBUILD_FLAGS ?=

PROFILE_SERVICE_cuda := worker
PROFILE_SERVICE_cpu := worker-cpu
PROFILE_SERVICE_rocm := worker-rocm
PROFILE_SERVICE_xpu := worker-xpu

.PHONY: help setup config gpu-list \
	preflight preflight-cuda preflight-rocm preflight-xpu \
	build build-cuda \
	up up-cuda up-cpu up-rocm up-xpu \
	rebuild rebuild-cuda rebuild-cpu rebuild-rocm rebuild-xpu \
	down logs test smoke-worker smoke-cpu smoke-cpu-transcription

help: ## Show available targets and configurable options
	@printf 'Usage: make <target> [PROFILE=cuda|cpu|rocm|xpu] [REBUILD_FLAGS=--no-cache]\n\n'
	@awk 'BEGIN {FS = ":.*## "; printf "Targets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create .env from .env.example when it does not exist
	@test -f .env || cp .env.example .env

config: ## Validate the resolved Compose configuration
	@$(COMPOSE) config --quiet

gpu-list: ## List detected GPUs and their compatible worker profiles
	@./scripts/list-compatible-gpus.sh

preflight: preflight-cuda ## Validate NVIDIA CUDA host and container access

preflight-cuda:
	@./scripts/preflight.sh

preflight-rocm: ## Validate AMD ROCm host device access
	@./scripts/preflight-rocm.sh

preflight-xpu: ## Validate Intel XPU host device access
	@./scripts/preflight-xpu.sh

build: ## Build the base application services
	@$(COMPOSE) build

build-cuda: ## Build the base services and CUDA worker
	@$(COMPOSE) --profile cuda build

up: up-$(PROFILE) ## Start PROFILE (default: cuda)

up-cuda: setup preflight-cuda ## Build and start the CUDA stack
	@$(COMPOSE) --profile cuda up -d --build

up-cpu: setup ## Build and start the CPU stack
	@$(COMPOSE) --profile cpu up -d --build

up-rocm: setup preflight-rocm ## Build and start the ROCm stack
	@$(COMPOSE) --profile rocm up -d --build

up-xpu: setup preflight-xpu ## Build and start the Intel XPU stack
	@$(COMPOSE) --profile xpu up -d --build

define rebuild_profile
	@$(COMPOSE) --profile $(1) build $(REBUILD_FLAGS) frontend backend $(PROFILE_SERVICE_$(1))
	@$(COMPOSE) --profile $(1) up -d --force-recreate frontend backend $(PROFILE_SERVICE_$(1))
endef

rebuild: rebuild-$(PROFILE) ## Rebuild and recreate PROFILE (default: cuda)

rebuild-cuda: setup preflight-cuda ## Rebuild and recreate app services plus the CUDA worker
	$(call rebuild_profile,cuda)

rebuild-cpu: setup ## Rebuild and recreate app services plus the CPU worker
	$(call rebuild_profile,cpu)

rebuild-rocm: setup preflight-rocm ## Rebuild and recreate app services plus the ROCm worker
	$(call rebuild_profile,rocm)

rebuild-xpu: setup preflight-xpu ## Rebuild and recreate app services plus the Intel XPU worker
	$(call rebuild_profile,xpu)

down: ## Stop containers while preserving named volumes
	@$(COMPOSE) down

logs: ## Follow frontend and backend logs
	@$(COMPOSE) logs -f backend frontend

test: ## Run backend tests, frontend build/type checks, and Compose validation
	@$(COMPOSE) --profile test build backend-test frontend
	@$(COMPOSE) --profile test run --rm backend-test
	@$(COMPOSE) config --quiet

smoke-worker: ## Check CUDA worker dependencies without GPU access
	@$(COMPOSE) build worker
	@docker run --rm --entrypoint python whisper-docker-worker:latest -c "import torch, whisper; print(f'torch={torch.__version__} cuda={torch.version.cuda} visible={torch.cuda.is_available()}'); print(f'whisper={whisper.__file__}')"
	@docker run --rm --entrypoint ffmpeg whisper-docker-worker:latest -version

smoke-cpu: ## Check CPU worker dependencies
	@$(COMPOSE) --profile cpu build worker-cpu
	@docker run --rm --entrypoint python whisper-docker-worker-cpu:latest -c "import torch, whisper; assert not torch.cuda.is_available(); print(f'torch={torch.__version__} device=cpu'); print(f'whisper={whisper.__file__}')"

smoke-cpu-transcription: smoke-cpu ## Run a real one-second tiny.en CPU transcription
	@docker run --rm --entrypoint python whisper-docker-worker-cpu:latest /app/smoke/cpu_transcription.py
