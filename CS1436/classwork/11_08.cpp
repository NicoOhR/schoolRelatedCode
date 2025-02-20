#include <iostream>

using namespace std;
/*functions take 2D array representing the board and return the winner if found
 * 0 if undetermined. 
 *
 * isDraw iterates over the board and finds 0s, if none are found, the board must
 * be in a draw state.
 *
 * main uses the functions to return the state of the board.
 */


int checkHor(int board[][3]){
    for(int i = 0; i < 3; i++){
        if(board[i][0] == board[i][1] && board[i][0] == board[i][2]){
            return board[i][0];
        }
    }
    return 0;
}

int checkVer(int board[][3]){
    for(int i = 0; i < 3; i++){
        if(board[0][i] == board[1][i] && board[0][i] == board[2][i]){
                return board[0][i];
            }
        }
    return 0;
}

int checkDiagonal(int board[][3]){
    if(board[0][0] == board[1][1] && board[0][0] == board[2][2]){
        return board[0][0];
    }else if(board[2][0] == board[1][1] && board[2][0] == board[0][2]){
        return board[2][0];
    }
    return 0;
}

bool isDraw(int board[][3]){
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            if(board[i][j] == 0){
                return false;
            }
        }   
    }
    return true;
}

bool endgame(int board[][3]){
    if(checkHor(board) == 1 || checkVer(board) == 1 || checkDiagonal(board) == 1){
        cout << "Player 1 wins!";
        return true;
    }
    if(checkHor(board) == 2 || checkVer(board) == 2 || checkDiagonal(board) == 2){
        cout << "Player 2 wins!";
        return true;
    }
    if(isDraw(board)){
        cout << "The game is a draw!";
        return true;
    }else{
        return false;
    }

}

void getPosition(int board[][3], int &player){
    int horPos, verPos;

    cin >> horPos >> verPos;

    while(board[horPos][verPos] != 0 || horPos > 2 || verPos > 2){
        cout << "Illegal move!" << endl;
        cin >> horPos >> verPos;
    }
    
    board[horPos][verPos] = player;
    player = player == 1 ? 2 : 1;
}

void displayBoard(int board[][3]){
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            cout << board[i][j] << " ";
        }
        cout << endl;
    }
}

int main(){
    int board[][3] = {{0,0,0},{0,0,0},{0,0,0}};
    int playerTurn = 1;
    while(!endgame(board)){
       cout << endl;
       displayBoard(board);
       cout << endl;
       getPosition(board, playerTurn);
    };

    return 0;
}
