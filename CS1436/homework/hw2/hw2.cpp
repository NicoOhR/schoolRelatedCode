#include <iostream>
#include <cmath>

using namespace std;

int main(){

    double lightbulb, lightbulbHours, acPower, acHours, fans, fanHours, priceElectricity;

    double electricityUse, bulbsPercent, acPercent, fanPercent, bill;

    cout << "# of light bulbs:";
    cin >> lightbulb;

    cout << "Average # of hours each bulb is ON in a day:";
    cin >> lightbulbHours;

    cout << "AC unit's power:";
    cin >> acPower;

    cout << "Typical # of hours AC unit is ON in a day:";
    cin >> acHours;

    cout << "# of FANs: ";
    cin >> fans;

    cout << "Average # of hours each Fan is ON in a day: ";
    cin >> fanHours;

    cout << "Per-unit price of electricity: ";
    cin >> priceElectricity;

    electricityUse = (lightbulb * lightbulbHours * 60 * 30) + (acPower * acHours * 30) + (fans * fanHours * 40 *30); 

    bulbsPercent = (lightbulb * lightbulbHours * 60 * 30)/electricityUse;

    acPercent = (acPower * acHours * 30)/electricityUse;

    fanPercent = (fans * fanHours * 40 * 30)/electricityUse;

    bill = electricityUse * priceElectricity;

    cout << "Total electricity usage: " << ceil(electricityUse/1000) << " kWh" << endl;

    cout << "Bulbs: " << bulbsPercent << "%  " << "AC: " << acPercent << "%  " << "FANs: " << fanPercent << "%" << endl;

    cout << "Electricity bill for the month: $  " << bill <<endl;

}
