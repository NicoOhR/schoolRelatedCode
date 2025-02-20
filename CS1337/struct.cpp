#include <array>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

int numPeople = 0;

struct Person {
  int age;
  string firstName, lastName;
  double income;
};

Person *persons;

int findSmallest(int startIndex) {
  int smallestValue = persons[startIndex].age;
  int smallestIndex = startIndex;

  for (int i = smallestIndex + 1; i < numPeople; i++) {
    if (persons[i].age < smallestValue) {
      smallestValue = persons[i].age;
      smallestIndex = i;
    }
  }
  return smallestIndex;
}

void selectionSort() {
  for (int i = 0; i < numPeople; i++) {
    int smallestIndex = findSmallest(i);

    Person temp = persons[smallestIndex];
    persons[smallestIndex] = persons[i];
    persons[i] = temp;
  }
}

int main() {
  ifstream finput("group.txt");
  finput >> numPeople;

  persons = new Person[numPeople];

  for (int i = 0; i < numPeople; i++) {
    finput >> persons[i].age >> persons[i].firstName >> persons[i].lastName >>
        persons[i].income;
  }

  finput.close();

  cout << "Before sorting" << endl;

  for (int i = 0; i < numPeople; i++) {
    cout << persons[i].age << " " << persons[i].firstName << " "
         << persons[i].lastName << " " << persons[i].income;
  }

  cout << "After sorting" << endl;

  for (int i = 0; i < numPeople; i++) {
    cout << persons[i].age << " " << persons[i].firstName << " "
         << persons[i].lastName << " " << persons[i].income;
  }
}
