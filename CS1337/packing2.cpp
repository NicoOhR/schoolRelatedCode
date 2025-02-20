#include <iostream>

using namespace std;

int main() {
  int capacity, total = 0, current;
  cout << "Enter capacity: ";
  cin >> capacity;

  cout << "Input items" << endl;

  while (true) {
    cin >> current;
    total += current;
    if (total > capacity) {
      cout << "Total: " << total - current << endl;
      total = current;
    }
  }
}
