#include <iostream>

using namespace std;


int main(){
    int counter = 0;
    int passingTotal = 0;
    int totalStudents;
    double studentGrade;
    
    cout << "Please enter total students in class: ";
    cin >> totalStudents;

    while(counter < totalStudents){
        cout << "Please enter the student's grade: ";
        cin >> studentGrade; 
        if(studentGrade >= 60){
            passingTotal++;
        }
        counter++;
    }
    cout << "There are " << passingTotal << " passing students" << endl;
    return 0;
}
