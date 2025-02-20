#include <iostream>

using namespace std;

double computePrintRetail(double value = 0, double percent = 0){
    return value * (1+(percent)/100);
}

double getValue(){
    double value;
    cin >> value;
    if(value < 0){
        throw string("Negative value");
    }else{
        return value;
    }
}

double getMarkup(){
    double markup;
    cin >> markup;
    if(markup < 0){
        throw string("Negative value");
    }else{
        return markup;
    }
}

int main(){
    double value, percent; 
    try{
        value = getValue();
    }catch(string std){
        cout << "value cannot be negative!" << endl;
        return -1;
    }

    try{
        percent = getMarkup();
    }catch(string std){
        cout << "markup cannot be negative!" << endl;
        return -1;
    }
    
    cout << computePrintRetail(value, percent);
    return 0;
}
