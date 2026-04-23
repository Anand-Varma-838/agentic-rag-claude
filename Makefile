.PHONY: install ingest ingest-reset run test lint clean

install:
	pip install -r requirements.txt

ingest:
	python ingest.py --docs data/docs/

ingest-reset:
	python ingest.py --docs data/docs/ --reset

run:
	streamlit run app.py

test:
	pytest tests/ -v

lint:
	python -m py_compile src/agent.py src/retriever.py src/vectorstore.py src/tools.py src/tracer.py ingest.py app.py
	@echo "All files OK"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf data/chroma
