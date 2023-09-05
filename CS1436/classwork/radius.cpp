#include <iostream>

using namespace std;

int main(){
    const double PI = 3.14;
    double radius;

    radius = 5;

    double area = PI * (radius * radius);

    double perimeter = 3 * PI * radius;

    cout << "Area = " << area << endl;

    cout << "Perimeter = " << perimeter << endl;

    return 0;
}
