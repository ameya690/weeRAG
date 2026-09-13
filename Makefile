.PHONY: bench test lint

bench:
	python bench/run.py --dataset hotpotqa --subset 500 --seed 42

test:
	pytest -v

lint:
	ruff check wee/
