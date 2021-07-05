/******************************************************************************

                              Online C++ Compiler.
               Code, Compile, Run and Debug C++ program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <ctime>

//project details

#define N 7
#define maxNum (1<<N)

//network details

#define numlayers 4
#define maxNodes 21

//training deatils

#define learnRate 0.01

using namespace std;

ofstream fout ("add.out");
ifstream netIn ("add_net.in");
ofstream netOut ("add_net.out");

int numNodes[numlayers+1] = {14,20,20,14,8};

class NeuralNetwork{
public:
    double weights[numlayers][maxNodes][maxNodes];
	double bias[numlayers][maxNodes];
	double activation[numlayers+1][maxNodes];
	double inter[numlayers][maxNodes];
	
	double expected[maxNodes];
	double Dbias[numlayers][maxNodes];
	double Dactivation[numlayers+1][maxNodes];
    
    void randomize(){
        int l,i,j;
    	for(l=0; l<numlayers; l++){
    		for(i=0; i<numNodes[l]; i++){
    			for(j=0; j<numNodes[l+1]; j++){
    				weights[l][i][j] = randVal();
    			}
    		}
    	}
    	for(l=0; l<numlayers; l++){
    		for(i=0; i<numNodes[l+1]; i++){
    			bias[l][i] = randVal();
    		}
    	}
    }
    
    void pass(){
        int l,i,j;
        for(l=0; l<numlayers; l++){
    		for(i=0; i<numNodes[l+1]; i++){
    			inter[l][i] = bias[l][i];
    			for(j=0; j<numNodes[l]; j++){
    				inter[l][i] += weights[l][j][i] * activation[l][j];
    			}
    			activation[l+1][i] = nonlinear(inter[l][i]);
    		}
    	}
    }
    
    void train(){
    	pass();
    	int l,i,j;
    	for(i=0; i<numNodes[numlayers]; i++){
    	    Dactivation[numlayers][i] = 2 * (activation[numlayers][i] - expected[i]);
    	}
    	for(l=numlayers-1; l>=0; l--){
    		for(i=0; i<numNodes[l+1]; i++){
    			Dbias[l][i] = Dactivation[l+1][i] * dnonlinear(inter[l][i]);
    			bias[l][i] -= Dbias[l][i] * learnRate;
    		}
    		for(i=0; i<numNodes[l]; i++){
    			Dactivation[l][i] = 0;
    			for(j=0; j<numNodes[l+1]; j++){
    				Dactivation[l][i] += Dbias[l][j] * weights[l][i][j];
    			}
    		}
    		for(i=0; i<numNodes[l+1]; i++){
    			for(j=0; j<numNodes[l]; j++){
    				weights[l][j][i] -= Dbias[l][i] * activation[l][j] * learnRate;
    			}
    		}
    	}
	}
    
    void saveNet(){
        int l,i,j;
    	for(l=0; l<numlayers; l++){
    		for(i=0; i<numNodes[l]; i++){
    			for(j=0; j<numNodes[l+1]; j++){
    				netOut<<weights[l][i][j]<<' ';
    			}
    			netOut<<'\n';
    		}
    		netOut<<'\n';
    	}
    	netOut<<'\n';
    	for(l=0; l<numlayers; l++){
    		for(i=0; i<numNodes[l+1]; i++){
    			netOut<<bias[l][i]<<' ';
    		}
    		netOut<<'\n';
    	}
    }
    
    void readNet(){
        int l,i,j;
    	for(l=0; l<numlayers; l++){
    		for(i=0; i<numNodes[l]; i++){
    			for(j=0; j<numNodes[l+1]; j++){
    				netIn>>weights[l][i][j];
    			}
    		}
    	}
    	for(l=0; l<numlayers; l++){
    		for(i=0; i<numNodes[l+1]; i++){
    			netIn>>bias[l][i];
    		}
    	}
    }
    
    double randVal(){
    	return ((double)rand() / RAND_MAX)*2-1;
    }
    /*
    
    //ReLU function:
    
    double nonlinear(double x){
    	return (x>0)*x;
    }
    
    double dnonlinear(double x){
    	return (x>0);
    }
    
    */
	
    /*
    
    //Sigmoid function:
    
    double nonlinear(double x){
    	return 1/(1+exp(-x));
    }
    
    double dnonlinear(double x){
    	return nonlinear(x) * (1-nonlinear(x));
    }
    
	*/
    
    
    
    //Negative sigmoid function:
    
    double sigmoid(double x){
    	return 1/(1+exp(-x));
	}
	
	double nonlinear(double x){
		return 1 - sigmoid(x);
	}
	
	double dnonlinear(double x){
		return -sigmoid(x) * (1-sigmoid(x));
	}
	
	
	
	/*
	
	//Tanh function:
	
	double nonlinear(double x){
		return tanh(x);
	}
	
	double dnonlinear(double x){
		return 1 - tanh(x)*tanh(x);
	}
	
	*/
};

int binary[N+1];
NeuralNetwork net1;

void convertBinary(int x){
	for(int i=0; i<N+1; i++){
		binary[i] = x%2;
		x /= 2;
	}
}

double squ(double x){
	return x*x;
}

void evaluate(){
	int networkBit;
	bool correct;
	int score = 0;
	double error = 0;
	
	int i,j;
	int num1,num2;
	double value;
	for(i=0; i<16384; i++){
		num1 = i/128;
		num2 = i%128;
		convertBinary(num1);
		for(j=0; j<N; j++){
			net1.activation[0][j] = binary[j];
			//fout<<binary[j];
		}
		//fout<<' ';
		convertBinary(num2);
		for(j=0; j<N; j++){
			net1.activation[0][N+j] = binary[j];
			//fout<<binary[j];
		}
		//fout<<' ';
		convertBinary(num1 + num2);/*
		for(j=0; j<N+1; j++){
			fout<<binary[j];
		}
		fout<<' ';*/
		net1.pass();
		correct = true;
		for(j=0; j<N+1; j++){
			value = net1.activation[numlayers][j];
			if(value < 0.5){
				networkBit = 0;
			}
			else{
				networkBit = 1;
			}
			//fout<<networkBit;
			if(networkBit != binary[j]){
				correct = false;
			}
			error += squ(value - binary[j]);
		}
		if(correct){
			score++;
		}
		//fout<<'\n';
	}
	cout<<"Score: "<<score<<'\n';
	cout<<"Error: "<<error<<'\n';
}

int main(){
	srand(time(0));
	net1.randomize();
	
	cout<<"Training\n";
	
	int num1,num2;
	int i,j;
	for(i=0; i<2000000; i++){
		if(i%100000 == 0){
			cout<<i<<'\n';
			evaluate();
		}
		num1 = rand()%maxNum;
		num2 = rand()%maxNum;
		convertBinary(num1);
		for(j=0; j<N; j++){
			net1.activation[0][j] = binary[j];
		}
		convertBinary(num2);
		for(j=0; j<N; j++){
			net1.activation[0][N+j] = binary[j];
		}
		convertBinary(num1 + num2);
		for(j=0; j<N+1; j++){
			net1.expected[j] = binary[j];
		}
		net1.train();
	}
	
	evaluate();
	
	net1.saveNet();
	
}
