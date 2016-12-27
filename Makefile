CURRENT_PATH := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

all:init

init:
	CURRENT_PATH=$(CURRENT_PATH) bash $(CURRENT_PATH)/tools/init/init.sh

clean:
	CURRENT_PATH=$(CURRENT_PATH) bash $(CURRENT_PATH)/tools/clean/clean.sh
