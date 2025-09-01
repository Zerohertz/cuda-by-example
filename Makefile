SHELL := /bin/zsh
CONTAINER_NAME := gpu
NVCC := nvcc
NVCC_FLAGS := -x cu -std=c++20 -gencode arch=compute_90,code=sm_90

.DEFAULT_GOAL := help
.PHONY: docker exec stop rm run compile clean bear help

# Prevent make from trying to build the file arguments as targets
$(FILE_ARG):
	@:
	
# Get file from command line arguments (everything after first target)
FILE_ARG = $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
FILE_PATH = $(firstword $(FILE_ARG))

# Docker management
docker:
	docker run \
	-d --name $(CONTAINER_NAME) \
	-v ./:/workspace \
	-w /workspace \
	zerohertzkr/gpu

exec:
	docker exec -it $(CONTAINER_NAME) zsh

stop:
	docker stop $(CONTAINER_NAME)

rm:
	docker rm -f $(CONTAINER_NAME)

# Compile, run, and clean up CUDA file
# Usage: make run src/chapter03/01_hello_world.cu
run:
	@if [ -z "$(FILE_PATH)" ]; then echo "Usage: make run <path/to/file.cu>"; exit 1; fi
	$(NVCC) $(NVCC_FLAGS) -o $(basename $(FILE_PATH)) $(FILE_PATH)
	./$(basename $(FILE_PATH))
	rm $(basename $(FILE_PATH))

# Compile only
# Usage: make compile src/chapter03/01_hello_world.cu
compile:
	@if [ -z "$(FILE_PATH)" ]; then echo "Usage: make compile <path/to/file.cu>"; exit 1; fi
	$(NVCC) $(NVCC_FLAGS) -o $(basename $(FILE_PATH)) $(FILE_PATH)

# Clean up all compiled binaries
# Usage: make clean
clean:
	find ./src -type f -executable ! -name "*.cu" ! -name "*.cpp" ! -name "*.c" ! -name "*.h" ! -name "*.sh" ! -name "*.py" -delete

# Generate compile_commands.json with bear
# Usage: make bear src/chapter03/01_hello_world.cu
bear:
	@if [ -z "$(FILE_PATH)" ]; then echo "Usage: make bear <path/to/file.cu>"; exit 1; fi
	bear --append -- $(NVCC) $(NVCC_FLAGS) -o $(basename $(FILE_PATH)) $(FILE_PATH)

# Show help
help:
	@echo "Available targets:"
	@echo "  Docker commands:"
	@echo "    docker   - Start GPU-enabled Docker container"
	@echo "    exec     - Execute bash in running container"
	@echo "    stop     - Stop the container"
	@echo "    rm       - Remove the container"
	@echo ""
	@echo "  CUDA compilation:"
	@echo "    run      - Compile, run, and clean up CUDA file"
	@echo "    compile  - Compile CUDA file only"
	@echo "    clean    - Remove all compiled binaries"
	@echo "    bear     - Generate compile_commands.json with bear"
	@echo ""
	@echo "Usage: make <target> <path/to/file.cu>"
	@echo "Example: make run src/chapter03/01_hello_world.cu"
