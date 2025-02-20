#include <iostream>

using namespace std;

int main() {
  int arr[] = {130, 710, 670, 10, 230, 170, 50, 90, 330};

  int key, j;

  for (int i = 1; i < 9; i++) {
    key = arr[i];
    j = i - 1;

    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j];
      j = j - 1;
    }
    arr[j + 1] = key;

    for (int k = 0; k < 9; k++) {
      cout << arr[k] << " ";
    }
    cout << endl;
  }
}
