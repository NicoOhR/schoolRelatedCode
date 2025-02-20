#include <iostream>

using namespace std;

int main(){
    int employed, recentGrad;
    cin >> employed; 

    if(employed == 1){
        cin >> recentGrad;

        if(recentGrad == 1){
            cout << "you qualify";
        }else{
            cout << "you must have graduated in the past two years to qualify";
        }else{
            cout << "you must be employed to qualify";
        }
    }

}
