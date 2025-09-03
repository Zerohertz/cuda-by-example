# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a CUDA GPU programming study repository based on the book "CUDA by Example: An Introduction to General-Purpose GPU Programming". The codebase contains educational CUDA programs organized by book chapters.

## Common Commands

### Docker Setup (Recommended)
```bash
make docker        # Start GPU-enabled Docker container
make exec          # Execute zsh in running container
make init          # Initialize development environment with clang tools
make stop          # Stop the container
make rm            # Remove the container
```

### Building and Running
```bash
# CMake approach (for full builds)
make build         # Generate CMake build directory
cd build && make <target> && ./<target>

# Direct NVCC approach (for single files)
make run src/path/to/file.cu     # Compile, run, and clean up CUDA file
make compile src/path/to/file.cu # Compile CUDA file only
make clean         # Remove all compiled binaries
```

### Code Quality
```bash
make lint          # Run clang-tidy to fix code issues
make format        # Format code with clang-format
make bear src/path/to/file.cu # Generate compile_commands.json for IDE support
```

## Architecture and Structure

### Source Organization
- `src/chapter03/` - Basic CUDA concepts and device queries
- `src/chapter04/` - Parallel programming fundamentals (vector operations, Julia sets)
- `src/chapter05/` - Thread cooperation and dot products
- `src/include/` - Shared utilities and helper functions

### Key Components
- **handler.cuh** - CUDA error handling utilities and macros (`HANDLE_ERROR`, `HANDLE_NULL`)
- **cpu_bitmap.h** - CPU bitmap utilities and image processing functions
- **logger.h** - Logging utilities for debugging and information output
- **stb_image_write.h** - Image writing support (PNG, BMP, TGA, JPG formats)

### Build System Details
- **CMake**: Automatically discovers and builds all `.cpp` and `.cu` files in `src/`
- **CUDA Architecture**: Targets compute capability 9.0 (configurable via `CMAKE_CUDA_ARCHITECTURES`)
- **Compiler Support**: Uses Clang++ for both C++ and CUDA compilation
- **Standards**: C++17 and CUDA 17

### Development Environment
- Uses Docker with `zerohertzkr/gpu` image for GPU access
- Pre-commit hooks available for code quality
- Clang-based toolchain (clang-tidy, clang-format)
- NVCC flags: `-x cu -std c++17 --gpu-architecture compute_90 --gpu-code sm_90`

## Notes
- All CUDA files should include `src/include/handler.cuh` for error handling
- Use `HANDLE_ERROR()` macro for CUDA API calls
- The project follows a chapter-by-chapter learning structure from the CUDA by Example book
- Graphics-related examples may require OpenGL dependencies (handled by Docker environment)
