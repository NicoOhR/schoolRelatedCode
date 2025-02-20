#include <iostream>

using namespace std;

int main() {
  int capacity, total = 0, current;
  cout << "Enter capacity: ";
  cin >> capacity;

  cout << "Input items";

  while (total < capacity) {
    cin >> current;
    total += current;
  }
  cout << "Total: " << total - current << endl;
}
