#include <iostream>

using namespace std;

/* Create input variable userInput, counter variables negativeCount and positiveCount, finally create the sum and product variables sum and product
 * set all = 0 (other than negative product, which needs to be 1 to start with bc of math)
 *
 * prompt user for input, and store oin userInput
 *
 * enter loop, checking that the user input is not equal to 0 each time. Inside the loop, check if input is positive or negative, adjust counter and product or sum accordingly
 *
 * after exiting the loop, check if positive count is not equal to 0, if so check again if it equal to 1, if so, print accordingly, otherwise, print with different grammer. If the positive count is equal to 0, print messege saying so. 
 *
 * repeat reciprocally for the negative numbers
 */

int main(){

    int userInput, positiveCount = 0, negativeCount = 0, positiveSum = 0, negativeProduct = 1;
    
    cout << "Enter a whole number [enter 0 to end input]: ";
    
    cin >> userInput;
    
    //check if exit condition is met
    while(userInput != 0){
        if(userInput > 0){
            //if input is +, add to count and sum
            positiveCount++;
            positiveSum = positiveSum + userInput;
        }else{
            //otherwise, add to negative count and multiply product
            negativeCount++;
            negativeProduct *= userInput;
        }
        cout << "Enter another whole number [enter 0 to end input]: ";
        cin >> userInput;
    }
    
    cout << "\n";

    if(positiveCount != 0){
        if(positiveCount == 1){
            //if only one positive number, print correct grammer
            cout << positiveCount << " positive number was entered. It was a " << positiveSum << "." << endl;
        }else{
            //otherwise, print with plural grammer
            cout << positiveCount << " positive numbers were entered. Their sum was " << positiveSum << "." << endl;
        }
    }else{
        //if positiveCount is 0, inform user
        cout << "No positive numbers were entered." << endl;
    }
    
    cout << "\n";

    if(negativeCount != 0){
        if(negativeCount == 1){
            //if only one negative number, print singular grammer
            cout << negativeCount << " negative number was entered. It was a " << negativeProduct << "." << endl;       
        }else{
            //otherwise, print with plural grammer
            cout << negativeCount << " negative numbers were entered. Their product was " << negativeProduct << "." << endl;
        }
    }else{
        //if negativeCount is 0, inform user.
        cout << "No negative numbers were entered." << endl;
    }
}
