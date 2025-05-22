PYTHON ?= python3
LOCKFILE = .build.lock

all: $(LOCKFILE)

$(LOCKFILE): setup.py kernel.cu binding.cpp
	$(PYTHON) setup.py build_ext --inplace
	touch $(LOCKFILE)

run: $(LOCKFILE)
	$(PYTHON) test.py

clean:
	rm -rf build  __pycache__ my_kernel*.so $(LOCKFILE)

.PHONY: all run clean
