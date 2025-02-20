#include <iostream>

using namespace std;

//program to find the total revenue from a list of ages
//computePrice augments argArrPrice to store what each age paid
//compute total then iterates through the array, adding to the total
//overloaded function printValue takes either a double or int array and iterates through the print
//the values

void computePrice(int argArrSource[], double argArrPrice[], int argArrSize){
   
    double tktPrice[] = {0,8.0,12.0,20.0,12.0};
    
    for(int i = 0; i < argArrSize; i++){
        if(argArrSource[i] >= 0 && argArrSource[i] <= 5){
            argArrPrice[i] = tktPrice[0];
        }else if(argArrSource[i] >= 6 && argArrSource[i] <= 10){
            argArrPrice[i] = tktPrice[1];
        }else if(argArrSource[i] >= 11 && argArrSource[i] <= 18){
            argArrPrice[i] = tktPrice[2];
        }else if(argArrSource[i] >= 19 && argArrSource[i] <= 64){
            argArrPrice[i] = tktPrice[3];
        }else if(argArrSource[i] >= 65){
            argArrPrice[i] = tktPrice[4];
        }

    }
}

double computeTotal(double argArrSource[], int argArrSize){
    double total;

    for(int i = 0; i < argArrSize; i++){
        total += argArrSource[i];
    }

    return total;
}

void printValues(double argArrSource[], int argArrSize){
    for(int i = 0; i < argArrSize; i++){
        cout << argArrSource[i] << endl;
    }
}

void printValues(int argArrSource[], int argArrSize){
    for(int i = 0; i < argArrSize; i++){
        cout << argArrSource[i] << endl;
    }
}

int main(){ 
    int ages[] = {36,5,3,45,10,12,70,50,20,18};
    int arraySize = sizeof(ages)/sizeof(int);

    double prices[arraySize], total;

    computePrice(ages, prices, arraySize);
    
    total = computeTotal(prices, arraySize)
    
    printValues(prices, arraySize);

    return 0;

}

