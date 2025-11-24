CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I include
LDFLAGS =

# Directories
SRC_DIR = src
TEST_DIR = tests
EXAMPLE_DIR = examples
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES))

# Test files
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.cpp)
TEST_BINARIES = $(patsubst $(TEST_DIR)/%.cpp,$(BUILD_DIR)/%,$(TEST_SOURCES))

# Example files
EXAMPLE_SOURCES = $(wildcard $(EXAMPLE_DIR)/*.cpp)
EXAMPLE_BINARIES = $(patsubst $(EXAMPLE_DIR)/%.cpp,$(BUILD_DIR)/%,$(EXAMPLE_SOURCES))

# Default target
all: $(OBJECTS)

# Create build directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build tests
tests: $(TEST_BINARIES)

$(BUILD_DIR)/test_%: $(TEST_DIR)/test_%.cpp $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(OBJECTS) -o $@ $(LDFLAGS)

# Build examples
examples: $(EXAMPLE_BINARIES)

$(BUILD_DIR)/%: $(EXAMPLE_DIR)/%.cpp $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(OBJECTS) -o $@ $(LDFLAGS)

# Run tests
run-tests: tests
	@for test in $(TEST_BINARIES); do \
		echo "Running $$test..."; \
		$$test; \
	done

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all tests examples run-tests clean

# Help
help:
	@echo "Available targets:"
	@echo "  all         - Compile all source files (default)"
	@echo "  tests       - Build all tests"
	@echo "  examples    - Build all examples"
	@echo "  run-tests   - Build and run all tests"
	@echo "  clean       - Remove build artifacts"
	@echo "  help        - Show this help message"