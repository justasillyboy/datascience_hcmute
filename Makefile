PYTHON := python

.PHONY: all clean analysis mnar sensitivity figures model gridsearch test run submission

all: clean analysis mnar sensitivity figures model gridsearch test

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for root in ('src','tests','scripts') for p in pathlib.Path(root).rglob('__pycache__')]; shutil.rmtree('.pytest_cache', ignore_errors=True)"

analysis:
	$(PYTHON) -m src.evaluate.run_analysis

mnar:
	$(PYTHON) -m src.evaluate.mnar_sensitivity

sensitivity:
	$(PYTHON) -m src.evaluate.sensitivity

figures:
	$(PYTHON) -m src.viz.run_figures

model:
	$(PYTHON) -m src.models.run_modeling

gridsearch:
	$(PYTHON) -m src.models.run_grid_search

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) main.py

submission:
	$(PYTHON) scripts/make_submission.py
