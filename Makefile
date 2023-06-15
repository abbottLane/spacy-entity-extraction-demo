venv: venv/touchfile

venv/touchfile: requirements.txt
	test -d venv || python3 -m venv venv
	. venv/bin/activate; pip install -Ur requirements.txt; \
	python3 -m spacy download  en_core_web_trf; \
	pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
	touch venv/touchfile

test: venv
	. venv/bin/activate; \
	python spacy-llm-ner.py; \
	python spacy-ner.py

clean:
	rm -rf venv