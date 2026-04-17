SUBDIRS := debugger \
		   driver_api \
		   type_cast \
		   nvcc_macro \
		   pytorch \
		   uvm

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	@echo "Entering $@"
	$(MAKE) -C $@

clean: $(addprefix clean-,$(SUBDIRS))

$(addprefix clean-,$(SUBDIRS)):
	@dir=$(patsubst clean-%,%,$@); \
	echo "Cleaning $$dir"; \
	$(MAKE) -C $$dir clean