# 1. Compiler and Flags
CC = gcc
CFLAGS = -O3 -march=native -fPIC -shared -I../

# 2. Files
SRCS = tree.c treeMath.c api.c ../arena.c
HEADERS = tree.h ..//arena.h
TARGET = libtree.so
PYTHON_SCRIPT = main.py

# 3. Default Build Target
all: $(TARGET)

# 4. Compile the Shared Library
$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS)
	@echo "Successfully built $(TARGET)"

run: $(TARGET)
	@echo "Executing Python"
	python3 $(PYTHON_SCRIPT)

# 5. Cleanup
clean:
	rm -f $(TARGET)
	@echo "Cleaned up compiled binaries"

.PHONY: all clean
