#include <iostream>

using namespace std;

int main(){
    
    long int empId[7] = {5658842, 4520125, 7895122, 8777541, 8451277, 1302850, 7580489};
    int hours[7];

    double payRate[7];

    double wages[7];

    for(int i = 0; i < 7; i++){
        cout << "Please enter the pay rate for employee " << empId[i] << " ";

        cin >> payRate[i];

        while(payRate[i] < 15.0){
            cout << "Pay rate must be at least 15." << endl;
            cin >> payRate[i];
        }

        cout << "and hours worked? ";
    
        cin >> hours[i];

        while(hours[i] < 0){
            cout << "Hours must be positive." << endl;
            cin >> hours[i];
        }

        wages[i] = hours[i] * payRate[i];
    }

    for(int j = 0; j < 7; j++){
        cout << "Gross pay for " << empId[j] << " " << wages[j] << endl;
    }

}
