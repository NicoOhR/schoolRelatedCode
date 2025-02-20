#include <iostream>
#include <iomanip>
using namespace std;

void getOrder(int &spools, double &discount, char &shipping,double &shippingCharge){
    
    cout << "Please enter the number of spools ordered: ";
    cin >> spools;

    cout << "Enter the discount percentage for the customer: ";
    cin >> discount;

    cout << "Does the order include custom shipping and handling charges? [Enter Y for Yes or N for No]: ";
    cin >> shipping;

    if(tolower(shipping) == 'y'){
        cout << "Enter the shipping and handling charge:";
        cin >> shippingCharge;

    }else{
        shippingCharge = 15;
    }
}

void spoolsInStock(int &stock){
    cout << " Enter the number of spools in stock: ";
    cin >> stock;
}
double calculateSpoolCharges(int spools, double discount){
    return (spools * (134.95 * (1- (discount/100))));
}

void formatOutput(char shipping, int spools, int stock, double discount, double shippingCharge){
    double price;
    cout << endl;
    cout << "\t\tOrder Summary" << endl;
    cout << "==============================" << endl;
    if(spools < 1){
        cout << "Spools must be at least one." << endl;
        return;
    }
    if(discount < 0){
        cout << "The percentage cannot be negative." << endl;
        return;
    }
    if(shippingCharge < 0){     
        cout << "Shipping and handling cannot be negative.";
        return;
    }
    if(stock < 0){
        cout << "The number of spools cannot be negative." << endl;
        return;
    }
    
    if(tolower(shipping) != 'n' && tolower(shipping) != 'y'){
        cout << "The response should be Y for Yes or N for No." << endl;
        return;
    }

    if(stock == 0){
        cout << "Sorry, there are currently no spools in stock, ready to ship." << endl;
        return;
    }
    
    if(spools > stock){
        cout << spools - stock << " spools are on back order." <<endl;
        cout << stock << " spools are ready to ship." << endl;

        price = calculateSpoolCharges(stock, discount);
        
        if(discount > 0){
            cout << "The charges for " << stock << " spools (including a " << fixed << setprecision(1) << discount << "%" << " discount): $" << fixed << setprecision(2) << price << endl;
        }else{
            cout << "The charges for " << stock << " spools : $" << fixed << setprecision(2) << price << endl;
        }

        cout << "Shipping and handling for " << stock << " spools: $" << fixed << setprecision(2) << stock * shippingCharge  << endl;
        
        cout << "The total charges (incl. shipping & handling): $" << fixed << setprecision(2) << stock * shippingCharge + price << endl;

    }else{
        cout << spools << " spools are ready to ship." << endl;
        
        price = calculateSpoolCharges(spools,discount);
        
        if(discount > 0){
            cout << "The charges for " << spools << " spools (including a " << fixed << setprecision(1) << discount << "%" << " discount): $" << fixed << setprecision(2) << price << endl;
        }else{
            cout << "The charges for " << spools << " spools : $" << fixed << setprecision(2) << price << endl;
        }
        cout << "Shipping and handling for " << spools << " spools: $" << fixed << setprecision(2) << spools * shippingCharge << endl;

        cout << "The total charges (incl. shipping & handling): $" << fixed << setprecision(2) << spools * shippingCharge + price << endl; 
    }

    cout << endl <<  "Thank you, please shop again." << endl;

    
}

int main(){
    int spools, stock;
    double discount, shippingCharge;
    char shipping;

    getOrder(spools, discount, shipping, shippingCharge);
    spoolsInStock(stock);

    formatOutput(shipping, spools, stock, discount,shippingCharge);
}
