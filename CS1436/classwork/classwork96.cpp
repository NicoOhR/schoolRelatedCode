#include <iostream>
using namespace std;

int main(){

    int radius;

    float area, perimeter;

    cin >> radius;

    if(radius > 0){
        area = 3.14 * radius * radius;
        perimeter = 3.14 * radius * 2;
    }
    cout << area << endl << perimeter;

    return 0;
}
