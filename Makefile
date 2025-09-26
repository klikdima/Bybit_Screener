PY=python

venv:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:
	. .venv/bin/activate || ($(PY) -m venv .venv && . .venv/bin/activate); \
	pip install -r requirements.txt; \
	PYTHONPATH=src $(PY) -m bot.main
