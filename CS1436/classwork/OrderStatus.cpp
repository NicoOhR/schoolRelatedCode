#include <iostream>
#include <iomanip>
using namespace std;
/*
getOrder, returns nothing, takes refrences to spool, discount, shipping, and shippingCharge variables,
    *getOrder begins by inputting to the variable, and making sure that the inputted variable is valid for the catagory
    *if it is not, enter a while loop to inform the user and prompt them again
spoolsInStock, returns nothing, takes refrence to a stock variables
    *does a similar thing to getOrder, but specifically for the spools that are still in stock
calculateSpoolCharges, returns a double, takes int spools and double discount
    *the function will return the double amount of the charges, does not account for shipping and handling
formatOutput, returns nothing, takes spools, discount, shipping, and shippingCharge
    *formatOutput will calculate the final price, and output the order accordingly

*/


void getOrder(int &spools, double &discount, char &shipping,double &shippingCharge){
    
    cout << "Please enter the number of spools ordered: ";
    cin >> spools;

    while(spools < 1){
        cout << "Spools must be at least one." << endl;
        cin >> spools;
    }

    cout << "Enter the discount percentage for the customer: ";
    cin >> discount;
    while(discount < 0){
        cout << "The percentage cannot be negative." << endl;
        cin >> discount;
    }

    cout << "Does the order include custom shipping and handling charges? [Enter Y for Yes or N for No]: ";
    cin >> shipping;
    //tolower() is used to standardize the input, so the user can input either n or N, y or Y
    while(tolower(shipping) != 'n' && tolower(shipping) != 'y'){
        cout << endl;
        cout << "Error, invalid response. The response should be Y for Yes or N for No." << endl;
        cout << "Does the order include custom shipping and handling charges? [Enter Y for Yes or N for No]: ";
        cin >> shipping;
    }

    if(tolower(shipping) == 'y'){
        cout << "Enter the shipping and handling charge: ";
        cin >> shippingCharge;
        
        while(shippingCharge < 0){
            cout << endl; 
            cout << "Error, invalid charges entered. Shipping and handling cannot be negative." << endl;
            cout << "Enter the shipping and handling charge: ";
            cin >> shippingCharge;
        }
    }else{
        //defualt shipping charge if the user does not opt for a custom amount
        shippingCharge = 15;
    }    
}

void spoolsInStock(int &stock){
    cout << "Enter the number of spools in stock: ";
    cin >> stock;

    while(stock <= 0){
        cout << "The number of spools cannot be negative." << endl;
        cout << "Enter the number of spools in stock: ";
        cin >> stock;
    }
}
double calculateSpoolCharges(int spools, double discount){
    return (spools * (134.95 * (1- (discount/100))));
}

void formatOutput(int spools, int stock, double discount, double shippingCharge){
    double price;
    cout << endl;
    cout << "\t\tOrder Summary" << endl;
    cout << "==============================" << endl;

    if(stock == 0){
        cout << "Sorry, there are currently no spools in stock, ready to ship." << endl;
        return;
    }
    //if the spools are larger than the stock, the user will only recieve the pricing for the stock that is avilable to ship
    if(spools > stock){
        cout << spools - stock << " spools are on back order." <<endl;
        cout << stock << " spools are ready to ship." << endl;
        //price is calculated using the calculate spool charges
        price = calculateSpoolCharges(stock, discount);
        
        if(discount > 0){
            cout << "The charges for " << stock << " spools (including a " << fixed << setprecision(1) << discount << "%" << " discount): $" << fixed << setprecision(2) << price << endl;
        }else{
            cout << "The charges for " << stock << " spools : $" << fixed << setprecision(2) << price << endl;
        }

        cout << "Shipping and handling for " << stock << " spools: $" << fixed << setprecision(2) << stock * shippingCharge  << endl;
        
        cout << "The total charges (incl. shipping & handling): $" << fixed << setprecision(2) << stock * shippingCharge + price << endl;

    }else{
        //otherwise, the user will recieve all of the spools ordered
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
    //get the orders from the user
    getOrder(spools, discount, shipping, shippingCharge);
    //get the stock from the user
    spoolsInStock(stock);
    //output to the user a summary of their order
    formatOutput(spools, stock, discount,shippingCharge);
}
