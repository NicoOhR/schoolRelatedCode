#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

const int MAXFILES = 10;
const int MAXROWS = 1000;

// There are 2 distinct approaches for this problem:
// 1. Sequential processing - one file at a time. You need total[MAXROWS] array
// 2. Row-wise processing - open all input files. Read one value from each,
// total them and output.
//                          need an array of ifstream objects open for reading
//                          files in parallel.
// If you are excited, you can implement both!

// WRITE YOUR CODE HERE

int main() {
  string ui, filename;
  int value, index = 0, total = 0;
  int values[MAXFILES] = {0};

  getline(cin, ui);
  stringstream input;
  input << ui;

  while (input) {
    input >> filename;
    ifstream finput(filename);
    finput >> value;
    cout << value;
    total += value;
  }

  ofstream foutput("output.txt");
  for (int i = 0; i < MAXFILES; i++) {
    foutput << total << endl;
  }
  foutput.close();
}
