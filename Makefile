.PHONY: test pushs3

setup:
	pip install -rq requirements.txt
	pip install .

test:
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
