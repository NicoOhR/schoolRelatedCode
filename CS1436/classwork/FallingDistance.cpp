#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;
/*create function getSeconds:
 *  *getSeconds will prompt user for input, and make sure that it is at least 0
 *  * if the input is not at least 0, the function will enter a loop, prompting the user until valid input is added
 *  * the function will return a double of the user input
 *Two functions, findEarthFallDist and findMoonFallDist,both: input a double seconds, and calculates the fall distance on either the moon or the earth
 * a third, aux function inputs seconds, and moon and earth distance, allowing for easy formatting
 * in main, the program calls getSeconds and saves the input into double seconds
 * seconds, findEarthFallDist(seconds), and findMoonFallDist(seconds) are inputted into the outputFormat to print the end result
 *
 */


double getSeconds(){
    double seconds;

    cout << "Please enter the time of the fall (in seconds): ";
    cin >> seconds;

    while(seconds < 0){//if seconds is less than 0, inform user and take input again
        cout << "The time must be at least zero." << endl;
        cin >> seconds;
    }

    return seconds;
}

double findEarthFallDist(double seconds){
    const double g_Earth = 9.81; 

    return 0.5 * g_Earth * (seconds * seconds);
}

double findMoonFallDist(double seconds){
    const double g_Moon = 1.625;

    return 0.5 * g_Moon * (seconds * seconds);
}

void outputFormat(double seconds, double distanceEarth, double distanceMoon){
    cout << endl;
    cout << "The object traveled " << fixed << setprecision(3) << distanceEarth << " meters in " << setprecision(1) << (double)seconds << " seconds on Earth." << endl;

    cout << "The object traveled " << fixed << setprecision(3) << distanceMoon << " meters in " << setprecision(1) << (double)seconds << " seconds on the Moon." << endl;
}

int main(){
    
    double seconds;

    seconds = getSeconds();
    
    outputFormat(seconds, findEarthFallDist(seconds), findMoonFallDist(seconds));
}
