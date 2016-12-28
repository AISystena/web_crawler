MAKE_PATH := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

all:init

init:
	@MAKE_PATH=$(MAKE_PATH) bash $(MAKE_PATH)/tools/init/init.sh

clean:
	@MAKE_PATH=$(MAKE_PATH) bash $(MAKE_PATH)/tools/clean/clean.sh
