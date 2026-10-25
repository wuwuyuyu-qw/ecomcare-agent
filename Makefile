export PYTHONUTF8 = 1
export PYTHONPATH := $(CURDIR)
.PHONY: help offline-test init-env init-demo-auth dev kb-build kb-vectorize
help:
	@echo "offline-test / init-env / init-demo-auth / dev / kb-build / kb-vectorize"
offline-test:
	uv run --locked --group dev --no-group ml python -B scripts/check_agent_core.py --tests-only --compatible-course
init-env:
	uv run --locked python scripts/init_local_env.py
init-demo-auth:
	uv run --locked python scripts/init_demo_auth.py
dev:
	uv run --locked python scripts/run_local_agent.py --env-file "$(CURDIR)/.env"
kb-build:
	uv run --locked python scripts/build_kb.py
kb-vectorize:
	@echo "This command calls the embedding provider and may incur cost. Run manually if authorized."
	uv run --locked python scripts/vectorize_kb.py
