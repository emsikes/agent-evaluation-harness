.PHONY: test test-mock test-all run-triage run-adversarial

test:
	python -m pytest tests/test_eval.py -v

test-mock:
	python -m pytest tests/test_mock.py -v

test-all:
	python -m pytest tests/ -v

run-triage:
	python tests/run_dataset.py

run-adversarial:
	python tests/run_dataset.py