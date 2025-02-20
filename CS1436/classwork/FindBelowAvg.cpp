#include <iostream>
#include <iomanip>
using namespace std;

int main(){
    
    int grades[120], counter = 0;
    double average;


    for(int i = 0; i < 120; i++){
        cout << "Enter the first score or -999 to end input:";

        cin >> grades[i];

        if(grades[i] == -999){
            if(i == 0){
                cout << "No scores were entered." << endl;
                return 0;
            }
            break;
        }

        counter++;
    }

    for(int j = 0; j < counter; j++){
        average += grades[j];
    }

    average = average/(counter);

    cout << "The average of the scores is: " << fixed << setprecision(1) << average << "." << endl;
    
    if(counter != 1){
        cout << "The scores below the average were: ";

        for(int k = 0; k < counter; k++){
            if(grades[k] < average){
                cout << grades[k] << " ";
            }
        }

        cout << endl;
    }

    return 0;
}
