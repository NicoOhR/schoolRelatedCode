#include <iostream>

using namespace std;

//create an empty input variable
//input a number from the user
//check if the remainder of the input number, when divided by 2, is equal to 0
//if it is, print that the number is even, and if it is not, print that the number is odd

int main(){
    int input;

    cin >> input;
    //the % findes the reminder and we compare it to 0 to find the eveness of the number
    if(input % 2 == 0){
        cout << "The number is even \n";
    }else{
        cout << "The numebr is odd \n";
    }
}
