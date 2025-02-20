#include <cmath>
#include <iostream>

using namespace std;

struct Room {
  double length, width, height;
};

struct Floor {
  int floorNum, numRooms;
  Room *rooms;
};

double computeCA(Floor *floorPtr, int size) {
  double area = 0;
  for (int i = 0; i < floorPtr->numRooms; i++) {
    area += (floorPtr->rooms[i].length * floorPtr->rooms[i].width);
  }
  return area;
}

double computeWP(Floor *floorPtr, int size) {
  double area = 0;
  for (int i = 0; i < floorPtr->numRooms; i++) {
    area += (floorPtr->rooms[i].length * floorPtr->rooms[i].height +
             floorPtr->rooms[i].width * floorPtr->rooms[i].height) *
            2;
  }
  return area;
}

int main() {
  int numFloors, floorNum, numRooms;
  double l, w, h;
  cout << "Enter number of floors" << endl;
  cin >> numFloors;

  Floor allFloors[numFloors];

  for (int i = 0; i < numFloors; i++) {
    cout << "Enter the floor number and number of rooms for " << i << endl;
    cin >> floorNum >> numRooms;

    allFloors[i].floorNum = floorNum;
    allFloors[i].numRooms = numRooms;
    allFloors[i].rooms = new Room[allFloors[i].numRooms];

    for (int j = 0; j < numRooms; j++) {
      cout << "Enter length, width, and height" << endl;
      cin >> l >> w >> h;
      allFloors[i].rooms[j].length = l;
      allFloors[i].rooms[j].width = w;
      allFloors[i].rooms[j].height = h;
    }
  }

  cout << computeCA(&allFloors[0], 5) << endl;
}
