.PHONY: test pushs3

setup:
	pip install -q -r requirements.txt
	pip install .

test: setup
	pytest tests -m "not pybox2d" --tb=line --disable-pytest-warnings

test-with-pybox2d:
	pytest tests --tb=line --disable-pytest-warnings

tensorboard:
	tensorboard --logdir experiments

monitor:
	jupyter lab

pulls3:
	make pulls3-dataset
	make pulls3-nem

pulls3-dataset:
	aws s3 cp s3://energy-py/public/dataset.zip dataset.zip
	unzip dataset.zip

pulls3-nem:
	aws s3 cp s3://energy-py/public/nem.zip nem.zip
	unzip nem.zip; mv nem-data ~

setup-pybox2d-macos:
	brew install swig
	git clone https://github.com/pybox2d/pybox2d
	cd pybox2d_dev; python setup.py build; python setup.py install
