#include <iostream>
#include <iomanip>

using namespace std;

int main(){

    int bagles, coffee;
    double baglesCost, coffeeCost, subtotal, total;

    double const tax = 0.0825;
    cout << "Welcome to THE Bagel Store" << endl;

    cout << "How many bagels would you like? ";
    cin >> bagles;

    cout << "and how many cups of coffee? ";
    cin >> coffee;

    baglesCost = bagles * 1.99;

    coffeeCost = coffee * 1.20;

    subtotal = baglesCost + coffeeCost;

    total = subtotal * (1 + tax);

    cout << setw(10) << "Product" << setw(10) << "Price" << setw(10) << "Quantity" << setw(10) << "Cost" << endl;
    cout << setw(10) << "Coffee" << setw(10) << "1.20" << setw(10) << coffee << setw(10) << coffeeCost << fixed << setprecision(2) << endl;
    cout << setw(10) << "Bagel" << setw(10) << "1.99" << setw(10) << bagles << setw(10) << baglesCost << fixed << setprecision(2) << endl;
    cout << setw(30) << "Sub Total" << setw(10) << fixed << setprecision(2) << subtotal << endl;
    cout << setw(30) << "Tax" << setw(10) << fixed << setprecision(2) << subtotal * tax << endl;
    cout << setw(30) << "Total" << setw(10) << fixed << setprecision(2) << total << endl;
}
