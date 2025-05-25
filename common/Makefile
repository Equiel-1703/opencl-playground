BIN_FOLDER := ./bin
SRC_FOLDER := ./src

CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17
LINKERFLAGS := -lOpenCL
SRC := $(wildcard $(SRC_FOLDER)/*.cpp)
OBJ := $(SRC:$(SRC_FOLDER)/%.cpp=$(BIN_FOLDER)/%.o)

all: $(BIN_FOLDER) $(OBJ)

$(BIN_FOLDER)/%.o: $(SRC_FOLDER)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LINKERFLAGS)

clean:
	rm -f $(OBJ)

$(BIN_FOLDER):
	mkdir -p $(BIN_FOLDER)

.PHONY: all clean