#include <iostream>

using namespace std;

void positive(double &price, double &markup){
    while(price < 0 || markup < 0){
        cout << "Please only enter positive numbers" << endl;
        cin >> price;
        cin >> markup;
    }
}

void computePrintRetail(double &price, double &markup){
    positive(price, markup);

    cout << price + (price *(markup/100)) << endl;
}

int main(){
    double markup, price;

    cin >> price; 
    cin >> markup;

    computePrintRetail(price, markup);
}
