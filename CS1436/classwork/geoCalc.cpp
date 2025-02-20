#include <iostream>

using namespace std; 

/* create choice variable
 * present user with a menu
 * input user choice and store in choice variable
 *
 * set up switch statement, comparing user choice to 1,2,3 and 4
 * if the user input is 1, calculate area of circle, 2 area of rect, 3 area of triangle, and 4 to quit
 *
 * in each case block,create vars and prompt user for necessary var
 * calculate the area and output to the user.
 */

int main(){
    int choice;
    //present menu
    cout << "What would you like to calculate? \n \t 1. Calculate Area of Circle \n \t 2. Calculate Area of Rectangle \n \t 3. Calculate Area of triangle \n \t 4. Quit \n Enter your choice <1-4>" << endl;

    cin >> choice;

    switch(choice){
        case 1:
            //prompt user for radius, calculate area and output
            double radius;
            cout << "Please enter radius: " << endl;
            cin >> radius;
            cout << "Area of circle: " << radius * radius * 3.14 << endl;
            break;
        case 2: 
            //prompt user for length and width, calculate are and output
            double length, width;
            cout << "Enter length: " << endl;
            cin >> length;
            cout << "Enter width: " << endl;
            cin >> width;
            cout << "Area of rectangle: " << width * length << endl;
            break;
        case 3:
            //prompt user for height and base length, calculate are and output
            double base, height;
            cout << "Enter base: " << endl;
            cin >> base;
            cout << "Enter height: " << endl;
            cin >> height;
            cout << "Area of triangle: " << (base * height)/2 << endl;
            break;
        case 4:
            //exit case
            cout << "Have a good day!" << endl;
            break;
    }
    return 0;

}
