.PHONY: test lint compile eval

test:
	PYTHONPATH=src python -m pytest tests -q -p no:cacheprovider

lint:
	python -m ruff check src tests

compile:
	python -m compileall src tests

eval:
	PYTHONPATH=src python -m travel_agent.rag.cli eval --embedding-provider local --json
