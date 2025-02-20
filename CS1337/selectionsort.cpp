#include <iostream>

using namespace std;

int main() {
  int arr[] = {130, 710, 670, 10, 230, 170, 50, 90, 330};

  int min, temp;

  for (int i = 0; i < 9; i++) {
    min = i;

    for (int j = i + 1; j < 9; j++) {
      if (arr[j] < arr[min]) {
        min = j;
      }
    }

    if (min != i) {
      temp = arr[min];
      arr[min] = arr[i];
      arr[i] = temp;
    }

    for (int k = 0; k < 9; k++) {
      cout << arr[k] << " ";
    }

    cout << endl;
  }
}
