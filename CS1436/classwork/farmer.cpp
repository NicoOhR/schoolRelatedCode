#include <iostream>

using namespace std;

int main(){
    float cost;

    cout << "How much are you buying?" << endl;

    cin >> cost;

    if(cost >= 200){
        cout << "Your total is " << cost * 0.85 << endl;
    }else if(cost >= 100){
        cout << "Your total is " << cost * 0.9 << endl;
    }else{
        cout << "Your total is " << cost << endl;
    }
    
    return 0;
}
