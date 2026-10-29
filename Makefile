.PHONY: all test
all:
	cd experiments && python 01_tail_costs.py && python 02_escalation_queue.py && python 03_autonomy_thresholds.py
test:
	pytest -q
