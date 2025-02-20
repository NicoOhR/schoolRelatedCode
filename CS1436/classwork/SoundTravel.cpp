#include <iostream>
#include <iomanip>
using namespace std;

/*Init an unsigned int userChoice (sign of input is irrelevant), an input double distance, and the constants of speed through the medium
 * present user with a menu and prompt for choice, storing in userChoice
 * init double speed and string medium
 * compare user choice with different cases, set speed and medium appropriatly, otherwise, return an error
 * prompt user for distance, and check if it is valid, if not, return error
 * output to the user the final output with the calculate time and inputted distance.
 */

int main(){
    unsigned int userChoice; 
    double distance;
    //speed constants 
    const double WOODSPD = 12631.23;
    const double STEELSPD = 10614.81;
    const double WATERSPD = 4714.57;
    const double AIRSPD = 1125.33;
    //menu options
    cout << "\tTime for Sound to Travel through a Medium given Distance\n\n1 - Wood\n2 - Steel\n3 - Water\n4 - Air\n\n";
    cout << "Enter the number of the medium: " << endl;
    //input the number user selected
    cin >> userChoice;

    double speed;
    string medium;
    
    //compare user choice to cases and set speed and medium appropriatly
    switch(userChoice){
        case 1:
            speed = WOODSPD;
            medium = "wood";
            break;
        case 2:
            speed = STEELSPD;
            medium = "steel";
            break;
        case 3:
            speed = WATERSPD;
            medium = "water";
            break;
        case 4:
            speed = AIRSPD;
            medium = "air";
            break;
        default:
            //all other cases invalid, cout an error and end the program
            cout << "Error, invalid entry!\nPlease run the program again." << endl;
            return 0;


    }
    //prompt user for distance  
    cout << "Enter the distance to travel (in feet): " << endl;
    cin >> distance;
   
    //ensure that distance is a valid number, and if not, return an error and end program 
    if(distance <= 0){
        cout << "Error, the distance must be greater than zero." << endl;
        return 0; 
    }
    //output final result and format doubles to correct precision.
    cout << "In " << medium << " it will take " << fixed << setprecision(4) << distance/speed << " seconds to travel " << fixed << setprecision(2) << distance << " feet." << endl;

}
