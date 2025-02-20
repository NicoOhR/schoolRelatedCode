#include <iostream>

using namespace std;

/*init long long int seconds
 * create long long ints for the seconds in minutes, hours, and days
 * before every print, check if the amount of seconds is equal to or greater than one of the next unit:
 *  if it is not, go to the next unit
 */


int main(){
    long long int seconds;

    //seconds in a minute
    long long int secondsInMinute = 60;
    //seconds in an hour
    long long int secondsInHours = 60 * secondsInMinute;
    //seconds in days 
    long long int secondsInDays = 24 * secondsInHours; 

    cout << "Enter a time in seconds: " << endl;
    cin >> seconds;
    cout << endl;
    //check that there are more than 0 seconds
    if(seconds <= 0){
        cout << "Error! The seconds must be greater than zero." << endl;
    }else{
    cout << seconds << " seconds is:" << endl;
    }
    //if there are less seconds than there are in a day, skip this print 
    if(seconds >= secondsInDays){
        cout << "\t" << seconds/secondsInDays << " days." << endl;
    }
    //store the reminder of seconds after all the days
    seconds %= secondsInDays;
    
    //if there are less seconds than there are in an hour, skip this print 
    if(seconds >= secondsInHours){
        cout << "\t" << seconds/secondsInHours << " hours." << endl;
    }
    //store the reminder of seconds after all the hours
    seconds %= secondsInHours;

    //if there are less seconds than there are in a minute, skip this print
    if(seconds >= secondsInMinute){
        cout << "\t" << seconds/secondsInMinute << " minutes." << endl;
    }

    //store the reminder of seconds after all the minutes
    seconds %= secondsInMinute;

    //if there are no more seconds, skip this print
    if(seconds > 0){
        cout << "\t" << seconds << " seconds." << endl;
    }
}
