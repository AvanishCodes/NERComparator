REQ = $(shell pip install -r requirements.txt)
COMPILE = $(shell python3 main.py)
all:
	$(REQ)
	$(COMPILE)
	echo "Completed building the file"