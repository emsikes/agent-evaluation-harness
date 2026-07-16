.PHONY: test run-triage run-adversarial

test:
	python -m pytest code/tests/test_eval.py -v

run-triage:
	python code/tests/runner_agent_test.py

run-adversarial:
	python code/tests/runner_agent_test.py
