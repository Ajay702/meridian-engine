.PHONY: install test lint typecheck eval bench serve docker

install:
	uv venv && uv pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check src tests benchmarks

typecheck:
	mypy src

eval:
	meridian eval examples/eval.jsonl --corpus examples/corpus \
	  --gate recall@k=0.9 --gate groundedness=0.8 --gate answer_match=0.8

bench:
	python benchmarks/bench.py

serve:
	meridian --config meridian.example.toml serve --port 8080

docker:
	docker compose up --build
