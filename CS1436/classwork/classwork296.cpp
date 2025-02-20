#include <iostream>

using namespace std;

int main(){
    int grade1, grade2, grade3;

    cin >> grade1 >> grade2 >> grade3;

    if(((grade1 + grade2 + grade3)/3) >= 95){
        cout << "Congragulations!";
    }

    return 0;
}
