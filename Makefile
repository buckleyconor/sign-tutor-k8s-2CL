.PHONY: install test unit lint format helm-template

install:
	pip install -r requirements-dev.txt

unit:
	pytest -m "not integration" -q

test: unit

integration:
	pytest -m integration -q

lint:
	ruff check src training tests
	black --check src training tests

format:
	black src training tests

helm-template:
	helm template tutor-p1 k8s/chart --set participant=p1 --set host=p1.lab.internal
