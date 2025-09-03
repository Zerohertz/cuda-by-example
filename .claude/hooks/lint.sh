#!/bin/bash

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.filePath // empty')

# Only process C++/CUDA files
if [[ ! "$FILE_PATH" =~ \.(cu|cpp|h)$ ]] || [[ ! -f "$FILE_PATH" ]]; then
	exit 0
fi

# Exit if lint is disabled
if [[ "$DISABLE_LINT" -eq 1 ]]; then
	exit 0
fi

echo "🔍 Running clang-tidy on $FILE_PATH..."

# Run clang-tidy on the specific file
LINT_SUCCESS=1

# Check if build directory exists for compile commands
if [[ -d "build" ]]; then
	# Capture clang-tidy output and errors
	LINT_OUTPUT=$(clang-tidy -p build --fix "$FILE_PATH" 2>&1)
	LINT_EXIT_CODE=$?

	if [[ $LINT_EXIT_CODE -ne 0 ]]; then
		echo "❌ clang-tidy failed on $FILE_PATH" >&2
		echo "Error details:" >&2
		echo "$LINT_OUTPUT" >&2
		LINT_SUCCESS=0
	fi
else
	echo "⚠️  Build directory not found. Run 'make build' first for better clang-tidy analysis."

	# Capture clang-tidy output and errors
	LINT_OUTPUT=$(clang-tidy --fix "$FILE_PATH" 2>&1)
	LINT_EXIT_CODE=$?

	if [[ $LINT_EXIT_CODE -ne 0 ]]; then
		echo "❌ clang-tidy failed on $FILE_PATH" >&2
		echo "Error details:" >&2
		echo "$LINT_OUTPUT" >&2
		LINT_SUCCESS=0
	fi
fi

if [[ "$LINT_SUCCESS" -eq 1 ]]; then
	echo "✅ Lint completed" >&2
else
	echo "❌ Lint failed" >&2
	exit 2
fi

exit 0
