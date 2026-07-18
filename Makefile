.PHONY: test run-triage run-adversarial

test:
	python -m pytest tests/test_eval.py -v

run-triage:
	python tests/runner_agent_test.py

run-adversarial:
	python tests/runner_agent_test.py