#include <cmath>
#include <iostream>

using namespace std;

int main() {
  double ones, twos, threes;
  cin >> ones >> twos >> threes;

  cout << endl << ceil((ones + twos * 2 + threes * 3) / 3);
}
