BUILD_DIR = build

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17
LINKFLAGS = -lOpenCL
TARGET = $(BUILD_DIR)/hello_world
SRC = $(wildcard *.cpp)

all: $(BUILD_DIR) $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LINKFLAGS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(TARGET)