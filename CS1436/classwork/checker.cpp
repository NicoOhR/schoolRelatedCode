#include <iostream>

using namespace std;

int main(){
    for(int i = 0; i<5;i++){
        if(i%2==0){
            cout << "0 1 0 1 0" << endl;
        }else{
            cout << "1 0 1 0 1" << endl;
        }
    }
}
