#include <iostream>

using namespace std;

int main(){

    int currentValue, highValue = 0, player,temp;

    for(int i = 1; i <= 4; i++){
        currentValue = 0;
        for(int j = 0; j<4; j++){
            cin >> temp;
            currentValue += temp;
        }
        if(currentValue > highValue){
            highValue = currentValue;
            player = i;
        }
    }
    cout << player << " " << highValue;
    return 0;
}
