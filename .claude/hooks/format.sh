#!/bin/bash

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.filePath // empty')

# Only process C++/CUDA files
if [[ ! "$FILE_PATH" =~ \.(cu|cpp|h)$ ]] || [[ ! -f "$FILE_PATH" ]]; then
	exit 0
fi

# Exit if format is disabled
if [[ "$DISABLE_FORMAT" -eq 1 ]]; then
	exit 0
fi

echo "🔧 Running clang-format on $FILE_PATH..."

# Run clang-format on the specific file
FORMAT_SUCCESS=1

# Capture clang-format output and errors
FORMAT_OUTPUT=$(clang-format -i "$FILE_PATH" 2>&1)
FORMAT_EXIT_CODE=$?

if [[ $FORMAT_EXIT_CODE -ne 0 ]]; then
	echo "❌ clang-format failed on $FILE_PATH" >&2
	echo "Error details:" >&2
	echo "$FORMAT_OUTPUT" >&2
	FORMAT_SUCCESS=0
fi

if [[ "$FORMAT_SUCCESS" -eq 1 ]]; then
	echo "✅ Format completed" >&2
else
	echo "❌ Format failed" >&2
	exit 2
fi

exit 0
