#include <iostream>

const int board[8][8] = {{0}};

using namespace std;

bool capturePawn(int argRow, int argColumn) {
  for (int i = 0; i < 8; i++) {
    if (board[argRow][i] == 1 || board[i][argColumn] == 1) {
      return true;
    }
  }
  return false;
}

int main() {}
