# ==========================================
# C-Engine Makefile
# ==========================================

# 1. Compiler and Flags
CC = gcc
CFLAGS = -O3 -march=native -ffast-math -fPIC -shared
LDFLAGS = -lopenblas -lm

# 2. Files
SRCS = engine.c mathlib.c arena.c api.c
HEADERS = engine.h mathlib.h arena.h
TARGET = libengine.so
PYTHON_SCRIPT = main.py

# 3. Default Build Target
all: $(TARGET)

# 4. Compile the Shared Library
# (Notice: This rule depends on your .c and .h files. If none have changed, it skips compilation!)
$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)
	@echo "Successfully built $(TARGET)"

# 5. Compile AND Run Python
run: $(TARGET)
	@echo "Launching Python Pipeline"
	python3 $(PYTHON_SCRIPT)

# 6. Cleanup
clean:
	rm -f $(TARGET) *.bin
	@echo "Cleaned up compiled binaries"

# Mark these targets as commands, not files
.PHONY: all run clean
