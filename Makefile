SHELL := /bin/zsh
CONTAINER_NAME := gpu
NVCC := nvcc
NVCC_FLAGS := -x cu -std c++17 --gpu-architecture compute_90 --gpu-code sm_90

.DEFAULT_GOAL := help
.PHONY: docker exec stop rm init lint format clang nvcc run compile clean bear help

# Prevent make from trying to build the file arguments as targets
$(FILE_ARG):
	@:
	
# Get file from command line arguments (everything after first target)
FILE_ARG = $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
FILE_PATH = $(firstword $(FILE_ARG))

# Docker container management
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

# Initialize development environment
init:
	echo 'export PATH=$${PATH}:/usr/local/cuda/bin' > ~/.env
	curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-archive-keyring.gpg \
		-o /usr/share/keyrings/cuda-archive-keyring.gpg
	sed -i '/signed-by/!s@^deb @deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] @' \
		/etc/apt/sources.list.d/cuda.list
	apt-get update && \
	apt-get install -y \
	gdb cuda-toolkit-12-6 \
	clang-tidy clang-format \
	libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev
	uv pip install pre-commit
	pre-commit install

# Code quality tools
lint:
	find src \
	-name "*.cpp" \
	-o -name "*.cu" \
	| xargs clang-tidy \
	-p build \
	--header-filter='.*' \
	--fix

format:
	find src \
	-name "*.cpp" \
	-o -name "*.h" \
	-o -name "*.cu" \
	-o -name "*.cuh" \
	| xargs clang-format -i

# Generate CMake build directory with Clang++
# Usage: make clang [DEBUG=1]
clang:
	rm -rf build && \
	cmake -S . -B build \
	-DUSE_NVCC=OFF \
	$(if $(DEBUG),-DENABLE_CUDA_DEBUG=ON)

# Generate CMake build directory with NVCC
# Usage: make nvcc [DEBUG=1]
nvcc:
	rm -rf build && \
	cmake -S . -B build \
	-DUSE_NVCC=ON \
	$(if $(DEBUG),-DENABLE_CUDA_DEBUG=ON)

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
	find ./src -type f -executable \
	! -name "*.c" \
	! -name "*.cpp" \
	! -name "*.h" \
	! -name "*.cu" \
	! -name "*.cuh" \
	! -name "*.sh" \
	! -name "*.py" \
	-delete

# Generate compile_commands.json for IDE support with bear
# Usage: make bear src/chapter03/01_hello_world.cu
bear:
	@if [ -z "$(FILE_PATH)" ]; then echo "Usage: make bear <path/to/file.cu>"; exit 1; fi
	bear --append -- $(NVCC) $(NVCC_FLAGS) -o $(basename $(FILE_PATH)) $(FILE_PATH)

# Show help
help:
	@echo "Available targets:"
	@echo "  Docker container management:"
	@echo "    docker   - Start GPU-enabled Docker container"
	@echo "    exec     - Execute zsh in running container"
	@echo "    stop     - Stop the container"
	@echo "    rm       - Remove the container"
	@echo ""
	@echo "  Development environment:"
	@echo "    init     - Initialize development environment with clang tools and pre-commit"
	@echo "    clang    - Generate CMake build directory (Clang++)"
	@echo "    nvcc     - Generate CMake build directory (NVCC)"
	@echo "    lint     - Run clang-tidy to fix code issues"
	@echo "    format   - Format code with clang-format"
	@echo ""
	@echo "  Debug mode: Add DEBUG=1 to any build command"
	@echo "    make clang DEBUG=1    - Clang++ with debug"
	@echo "    make nvcc DEBUG=1     - NVCC with debug"
	@echo ""
	@echo "  CUDA compilation:"
	@echo "    run      - Compile, run, and clean up CUDA file"
	@echo "    compile  - Compile CUDA file only"
	@echo "    clean    - Remove all compiled binaries"
	@echo "    bear     - Generate compile_commands.json for IDE support"
	@echo ""
	@echo "Usage: make <target> <path/to/file.cu>"
	@echo "Example: make run src/chapter03/01_hello_world.cu"
