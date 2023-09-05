#include <iostream>

using namespace std;


int main(){
    int items;
    string produceName;
    double price;
    char sign;

    cin >> items;
    getline(cin, produceName);
    cin >> price;
    cin >> sign;

    cout << items << produceName << price << sign;
}
