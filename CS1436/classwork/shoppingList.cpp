#include <iostream>

using namespace std;

int main(){
    double priceList[] = { 9.99, 2.99, 12.50,15.30,24.50,39.99,5.99,39.99,5.99,3.99,4.50,7.99,};

    double total = 0;
    
    for(int i = 0; i < 12; i++){
        if(total + priceList[i] > 50){
            cout << "Total " << total << endl;
            break;
        }
        cout << priceList[i] << " ";

        total = total + priceList[i];
    }

    total = 0.0;

    for(int i = 0; i < 12; i++){
        
        double price = priceList[i];

        if(total + price > 50){
           continue; 
        }
        cout << price << " ";

        total += price;
    }

    cout << "Total " << total << endl;
    
}
