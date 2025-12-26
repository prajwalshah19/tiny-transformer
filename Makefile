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

# Test runner (unified)
TEST_RUNNER = $(BUILD_DIR)/run_tests

# Individual test files
TEST_SOURCES = $(filter-out $(TEST_DIR)/run_tests.cpp, $(wildcard $(TEST_DIR)/test_*.cpp))
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

# Build unified test runner (single source of truth)
test: $(TEST_RUNNER)
	@$(TEST_RUNNER)

$(TEST_RUNNER): $(TEST_DIR)/run_tests.cpp $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(OBJECTS) -o $@ $(LDFLAGS)

# Build individual test executables (for debugging specific tests)
test-%: $(BUILD_DIR)/test_% 
	@$(BUILD_DIR)/test_$*

$(BUILD_DIR)/test_%: $(TEST_DIR)/test_%.cpp $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(OBJECTS) -o $@ $(LDFLAGS)

# Build examples
examples: $(EXAMPLE_BINARIES)

$(BUILD_DIR)/%: $(EXAMPLE_DIR)/%.cpp $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(OBJECTS) -o $@ $(LDFLAGS)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all test examples clean help test-%

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Compile all source files (default)"
	@echo "  test             - Build and run unified test suite"
	@echo "  test-tensor      - Build and run tensor tests only"
	@echo "  test-autograd    - Build and run autograd tests only"  
	@echo "  test-linear_regression - Build and run linear regression tests only"
	@echo "  examples         - Build all examples"
	@echo "  clean            - Remove build artifacts"
	@echo "  help             - Show this help message"
	@echo "  clean          - Remove build artifacts"
	@echo "  help           - Show this help message"