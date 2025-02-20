#include <iostream>

using namespace std;

/*
 * init doubles for weight, height, and bmi. Double is used to be precise
 * prompt user for weight, and then height, storing in height and weight
 * calculate bmi, and store in double
 * check if weight is smaller than or equal to 18.5, and tell user they are underweight
 * check if they are between the range of 18.5 and 25, and tell user they are optimal weight
 * otherwise, tell the user they are overweight
 *
 */

int main(){
    double weight, height, bmi;

    cout << "Please enter your weight: ";
    cin >> weight;
    cout << "Please enter your height: ";
    cin >> height;

    bmi = ((weight)/(height*height)) * 703; //calculate bmi through the formula

    if(bmi <= 18.5){//less than or equal to to make 18.5 included in underweight
        cout << "You are underweight";
    }else if(bmi > 18.5 && bmi < 25.0){//non-inclusive range between 18.5 and 25
        cout << "You are at an optimal weight";
    }else{//25 included in overweight
        cout << "You are overweight"  << endl;
    }
}
