BUILD_DIR = build

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17
LINKFLAGS = -lOpenCL
SRC = $(wildcard *.cpp)
BIN = $(SRC:%.cpp=$(BUILD_DIR)/%.o)
LINK_BINS = $(wildcard ../common/bin/*.o)

TARGET = $(BUILD_DIR)/hello_world.out

all: $(BUILD_DIR) $(BIN) $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(BIN) $(LINK_BINS) $(LINKFLAGS)

$(BUILD_DIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LINKFLAGS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(TARGET)
	rm -f $(BIN)