#include <iostream>

using namespace std;

int main(){
    
    int inputInts[5];
    int total = 0;

    for(int i = 0; i < 5; i++){
        cin >> inputInts[i];
        total += inputInts[i];
        cout << total << endl;
    }

    return 0;
}
