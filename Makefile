SUBDIRS := uvm \
	   driver_api \
	   debugger \
	   type_cast \
	   nvcc_macro \
	   pytorch

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

