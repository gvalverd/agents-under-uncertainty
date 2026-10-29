.PHONY: all case test
all: case
	cd experiments && python 01_tail_costs.py && python 02_escalation_queue.py && python 03_autonomy_thresholds.py
case:
	python case_study/contact_center.py && python case_study/plots.py
test:
	python -m pytest -q
