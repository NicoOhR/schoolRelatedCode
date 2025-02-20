#include <iostream>

using namespace std;
/*
 * Init int for the number user input, the positiveCounter, negative counter, total counter, the sum and the poduct. Create a user input character
 * by defualt, the user input charater should be y (for yes) the positive counter, negative counter, and total counter are all 0 at the start. Sum is 0 and product is 1 (otherwise it would never multiply correctly)
 * enter the loop and prompt user to input a number
 * evaluate the number, if it is positive, add to the positive counter and add it to the sum, if it is negative add to the negative counter and multiply the product by it, and if it is neither do nothing. In any case add to the total counter.
 * exit the loop when the user input is no longer 'y' 
 * if the positive counter is 0, tell the user as such, if it is 1 adjust grammer to make sense, otherwise, print out the number positive counter and sum with proper formatting, do the same for negative numbers and zeros
 */


int main(){
    
    int userNumber,positiveCounter = 0, negativeCounter = 0, sum = 0, totalCounter = 0, product = 1;
    char userInput = 'y';
  
    do{
        cout << "Enter whole number " << totalCounter + 1 << ": ";
        cin >> userNumber;
        totalCounter++;
        //if th number is positive, add to the counter and sum
        if(userNumber > 0){
           positiveCounter++;
           sum += userNumber;
           //otherwise if it is negative, add to the negative counter and multiply product
        }else if(userNumber < 0){
            negativeCounter++;
            product *= userNumber;
        }
        //prompt user to continue
        cout << "Would you like to enter another number?" << endl << "Enter Y for Yes or N for No: ";
        cin >> userInput;
    }while(tolower(userInput) == 'y');//tolower() so that the user could enter uppercase or lowercase 

    cout << endl;
    //adjust grammer to reflect how many positive values were entered
    if(positiveCounter == 0){
        cout << "No positive values were entered."; 
    }else if(positiveCounter == 1){
        cout << "One positive value was entered. It was a " << sum << ".";
    }else{
        cout << positiveCounter << " positive values were entered. Their sum was " << sum << ".";
    }

    cout << endl;
    //adjust grammer to reflect how many positive values were entered
    if(negativeCounter == 0){
        cout << "No negative values were entered.";
    }else if(negativeCounter == 1){
        cout << "One negative value was entered. It was a " << product << ".";
    }else{
        cout << negativeCounter << " negative values were entered. Their product was " << product << ".";
    }
    cout << endl; 

    //totalcounter - (positiveCounter + negativeCounter) is the amount of entries that were neither positive nor negative
    //and gives the amount of zeros the user entered
    if((totalCounter - (positiveCounter + negativeCounter)) == 0){
        cout << "No zeroes were entered.";
    }else if(totalCounter -(positiveCounter + negativeCounter) == 1){
        cout << "One zero was entered.";
    }else{
        cout << totalCounter - (positiveCounter + negativeCounter) << " zeroes were entered.";
    }
    cout << endl;
}

