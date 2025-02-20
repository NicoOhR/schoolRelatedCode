#include <iostream>
#include <sstream>
#include <string>

using namespace std;

int main() {

  int current, max = 0;
  string currentLine, token;

  getline(cin, currentLine);

  istringstream iss(currentLine);

  while (iss >> token) {
    if (istringstream(token) >> current) {
      if (current > max) {
        max = current;
      }
    } else {
      iss.clear();
    }
  }
  cout << max << endl;
}
