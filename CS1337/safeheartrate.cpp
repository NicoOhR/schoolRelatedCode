#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

int main() {
  // WRITE YOUR CODE HERE
  string input, file;
  double age, total = 0, safe = 0, line;

  getline(cin, input);

  istringstream iss(input);

  iss >> age >> file;

  ifstream dataFile(file);

  if (!dataFile.fail()) {
    while (dataFile >> line) {
      if (line < 220 - line) {
        safe += 1;
      }
      total += 1;
    }
  } else {
    cout << "Could not open file";
  }

  cout << safe / total << "%";
}
