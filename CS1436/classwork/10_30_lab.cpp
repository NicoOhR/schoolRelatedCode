#include <iostream>

using namespace std;
//Two arrays, one to store names of salsa and one to store the sales numbers
//int total to keep count of the total sold, highest and lowest for the indecies of the highest and lowest 
//selling salsas
//first loop for io, second to format output and find the highest and lowest selling

int main(){
    string heat[5] = {"mild", "medium", "sweet","hot","zesty"};

    int sales[5], total = 0, highest = 0, lowest = 0;

    for(int i = 0; i < 5; i++){
        cout << "Enter # of jars sold for " << heat[i] << ": ";
        cin >> sales[i];
    }

    for(int j = 0; j < 5; j++){
        cout << "# of jars " << heat[j] << " that was sold: " << sales[j] << endl;
        total += sales[j];
        if(sales[j] > sales[highest]){
            highest = j;
        }else if(sales[j] < sales[lowest]){
            lowest = j;
        }
    }

    cout << "Total sales : " << total << endl;

    cout << "Type with highest sale : " << heat[highest] << endl;

    cout << "Type with the lowest sale : " << heat[lowest] << endl;
}
