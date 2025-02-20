#include <iostream>

using namespace std;

int main(){
    int total;
    int userInput;

    cin >> userInput;

    while(userInput != -999){
        total += userInput;
        cin >> userInput;
    }

    cout << total << endl;
}
