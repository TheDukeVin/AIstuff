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

//environment details

#define numActions 4
#define numReactions 25
#define minScore 0
#define maxScore 1000

//network details

#define numlayers 2
#define maxNodes 6

//training deatils

#define numRollouts 1000
#define countRolls 500
#define learnRate 0.2
#define queueSize 5000
#define switchScore 6

#define batchSize 2

using namespace std;

ofstream fout ("5token.out");

int numNodes[numlayers+1] = {4,6,4};

class Agent{
public:
    double weights[numlayers][maxNodes][maxNodes];
	double bias[numlayers][maxNodes];
	double activation[numlayers+1][maxNodes];
	double inter[numlayers][maxNodes];
	
	double expected[maxNodes];
	double Dbias[numlayers][maxNodes];
	double Dweights[numlayers][maxNodes][maxNodes];
	double Dactivation[numlayers+1][maxNodes];
	
	double Sbias[numlayers][maxNodes];
	double Sweights[numlayers][maxNodes][maxNodes];
    
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
    
    void backProp(){
        pass();
    	int l,i,j;
    	for(i=0; i<numNodes[numlayers]; i++){
    	    Dactivation[numlayers][i] = 2 * (activation[numlayers][i] - expected[i]);
    	}
    	for(l=numlayers-1; l>=0; l--){
    		for(i=0; i<numNodes[l+1]; i++){
    			Dbias[l][i] = Dactivation[l+1][i] * dnonlinear(inter[l][i]);
    			for(j=0; j<numNodes[l]; j++){
    				Dweights[l][j][i] = Dbias[l][i] * activation[l][j];
    			}
    		}
    		for(i=0; i<numNodes[l]; i++){
    			Dactivation[l][i] = 0;
    			for(j=0; j<numNodes[l+1]; j++){
    				Dactivation[l][i] += Dbias[l][j] * weights[l][i][j];
    			}
    		}
    	}
    }
    
    void initS(){
		int l,i,j;
		for(l=0; l<numlayers; l++){
			for(i=0; i<numNodes[l]; i++){
				for(j=0; j<numNodes[l+1]; j++){
					Sweights[l][i][j] = 0;
				}
			}
		}
		for(l=0; l<numlayers; l++){
			for(i=0; i<numNodes[l+1]; i++){
				Sbias[l][i] = 0;
			}
		}
	}
	 
	void increment(){
		int l,i,j;
		for(l=0; l<numlayers; l++){
			for(i=0; i<numNodes[l]; i++){
				for(j=0; j<numNodes[l+1]; j++){
					Sweights[l][i][j] += Dweights[l][i][j];
				}
			}
		}
		for(l=0; l<numlayers; l++){
			for(i=0; i<numNodes[l+1]; i++){
				Sbias[l][i] += Dbias[l][i];
			}
		}
	}
	 
	void updateNet(){
		int l,i,j;
		for(l=0; l<numlayers; l++){
			for(i=0; i<numNodes[l]; i++){
				for(j=0; j<numNodes[l+1]; j++){
					weights[l][i][j] -= Sweights[l][i][j] / batchSize * learnRate;
				}
			}
		}
		for(l=0; l<numlayers; l++){
			for(i=0; i<numNodes[l+1]; i++){
				bias[l][i] -= Sbias[l][i] / batchSize * learnRate;
			}
		}
	}
    
    void print(){
        int l,i,j;
        cout<<"weights:\n";
    	for(l=0; l<numlayers; l++){
    		for(i=0; i<numNodes[l]; i++){
    			for(j=0; j<numNodes[l+1]; j++){
    				cout<<weights[l][i][j]<<' ';
    			}
    			cout<<'\n';
    		}
    		cout<<'\n';
    	}
    	cout<<"bias:\n";
    	for(l=0; l<numlayers; l++){
    		for(i=0; i<numNodes[l+1]; i++){
    			cout<<bias[l][i]<<' ';
    		}
    		cout<<'\n';
    	}
    }
    
private:
    double randVal(){
    	return ((double)rand() / RAND_MAX)*2-1;
    }
    
    double nonlinear(double x){
    	return 1/(1+exp(-x));
    }
    
    double dnonlinear(double x){
    	return nonlinear(x) * (1-nonlinear(x));
    }
};


int dir[4][2] = {{0,1}, {1,0}, {0,-1}, {-1,0}};

class Environment{
public:
    int timer;
    int score;
    int agentx,agenty;
    int tokenx,tokeny;
    const int boardx = 5;
    const int boardy = 5;
    
    void initialize(){
        timer = 0;
        score = 0;
        agentx = boardx/2;
        agenty = boardy/2;
        randToken();
    }
    
    int advance(int action){
        agentx += dir[action][0];
        agenty += dir[action][1];
        if(agentx == -1 || agentx == boardx){
            return -1;
        }
        if(agenty == -1 || agenty == boardy){
            return -1;
        }
        timer++;
        if(timer == 50){
        	return -1;
		}
        if(agentx == tokenx && agenty == tokeny){
            score++;
            randToken();
            return tokenx * boardy + tokeny;
        }
		return 0;
    }
    
    void inputAgent(Agent* a){
    	a->activation[0][0] = agentx;
    	a->activation[0][1] = agenty;
    	a->activation[0][2] = tokenx;
    	a->activation[0][3] = tokeny;
    }
    
    void copy(Environment* e){
        timer = e->timer;
        score = e->score;
        agentx = e->agentx;
        agenty = e->agenty;
        tokenx = e->tokenx;
        tokeny = e->tokeny;
    }
    
private:
    void randToken(){
        while(true){
            tokenx = rand()%boardx;
            tokeny = rand()%boardy;
            if(tokenx != agentx || tokeny != agenty){
                break;
            }
        }
    }
};


class data{
public:
    Environment e;
    double expected[numActions];
    
    void trainAgent(Agent* a){
        e.inputAgent(a);
        int i;
        for(i=0; i<numActions; i++){
            a->expected[i] = expected[i];
        }
        a->backProp();
        a->increment();
    }
    
    void print(){
        fout<<e.agentx<<' '<<e.agenty<<'\n';
        fout<<e.tokenx<<' '<<e.tokeny<<'\n';
        int i;
        for(i=0; i<numActions; i++){
            fout<<expected[i]<<' ';
        }
        fout<<'\n';
    }
};

class dataQueue{
public:
    data* queue[queueSize];
    int index = 0;
    
    void enqueue(data* d){
        queue[index%queueSize] = d;
        index++;
    }
    
    void trainAgent(Agent* a){
        int i,j;
        int trainIndex = 0;
		a->initS();
		for(i=0; i<min(index,queueSize); i++){
			queue[i]->trainAgent(a);
			trainIndex++;
			if(trainIndex == batchSize){
				trainIndex = 0;
				a->updateNet();
				a->initS();
			}
		}
    }
};

double propDist(double x){
    return x/(1-x);
}

int makeAction(Environment* e, Agent* a){
    /*
    e->inputAgent(a);
    a->pass();
    int maxIndex = 0;
    int i;
    for(i=1; i<numActions; i++){
        if(a->activation[numlayers][maxIndex] < a->activation[numlayers][i]){
            maxIndex = i;
        }
    }
    return maxIndex;
    */
    
    e->inputAgent(a);
    a->pass();
    
	double propSum = 0;
	int i;
	for(i=0; i<numActions; i++){
	    propSum += propDist(a->activation[numlayers][i]);
	}
	double parsum = 0;
	double randReal = (double)rand() / RAND_MAX * propSum;
	for(i=0; i<numActions; i++){
	    parsum += propDist(a->activation[numlayers][i]);
	    if(randReal <= parsum){
	        return i;
	    }
	}
	
}

void rollout(Environment* e, Agent* a){
    int action;
    while(true){
        action = makeAction(e,a);
        if(e->advance(action) == -1){
            break;
        }
    }
}

class searchNode{
public:
    Environment e;
    Agent* a;
    int numTries[numActions];
    int sumScore[numActions];
    int totalTries;
    searchNode* outcomes[numActions][numReactions];
    
    searchNode(Environment* copyEnv, Agent* copyAg){
        e.copy(copyEnv);
        a = copyAg;
        int i,j;
        for(i=0; i<numActions; i++){
            numTries[i] = 0;
            sumScore[i] = 0;
        }
        totalTries = 0;
        for(i=0; i<numActions; i++){
        	for(j=0; j<numReactions; j++){
        		outcomes[i][j] = NULL;
			}
		}
    }
    
    int searchPath(){ // return the score of this rollout.
        double maxPriority,currPriority;
        int maxIndex;
        maxPriority = minScore;
        int i;
        for(i=0; i<numActions; i++){
            if(numTries[i] == 0){
                maxPriority = maxScore;
                maxIndex = i;
                continue;
            }
            currPriority = (double)sumScore[i] / numTries[i] + 2*sqrt(log(totalTries)/numTries[i]);
            if(maxPriority < currPriority){
                maxPriority = currPriority;
                maxIndex = i;
            }
        }
        Environment copyEnv;
        copyEnv.copy(&e);
        int reaction = copyEnv.advance(maxIndex);
        int score;
        if(reaction != -1){
            if(outcomes[maxIndex][reaction] == NULL){
                outcomes[maxIndex][reaction] = new searchNode(&copyEnv,a);
                rollout(&copyEnv,a);
                score = copyEnv.score;
            }
            else{
                score = outcomes[maxIndex][reaction]->searchPath();
            }
        }
        else{
            score = copyEnv.score;
        }
        numTries[maxIndex]++;
        sumScore[maxIndex] += score;
        totalTries++;
        return score;
    }
};

class Trainer{
public:
    Agent a;
    dataQueue dq;
    bool certainMode = false;
    
    Trainer(){
        a.randomize();
    }
    
    void trainActions(Environment* e){
        searchNode* rootState = new searchNode(e,&a);
        int i;
        for(i=0; i<numRollouts; i++){
            rootState->searchPath();
        }
        // train the network
        data* newData = new data();
        newData->e.copy(e);
        for(i=0; i<numActions; i++){
            newData->expected[i] = (double)rootState->numTries[i] / numRollouts;
        }
        dq.enqueue(newData);
        newData->print(); // store the data
    }
    
    void trainGame(){
    	fout<<"New game:\n";
        Environment trainSpace;
        trainSpace.initialize();
        int action;
        while(true){
            trainActions(&trainSpace);
            action = makeAction(&trainSpace,&a);
            //action = randomAction(&trainSpace);
            if(trainSpace.advance(action) == -1){
                break;
            }
        }
        dq.trainAgent(&a);
    }
    
    int evaluate(){
        Environment evalSpace;
        evalSpace.initialize();
        int action;
        int i;
        while(true){
            action = makeAction(&evalSpace,&a);
            if(evalSpace.advance(action) == -1){
                break;
            }
        }
        return evalSpace.score;
    }
    
    double multEval(){
        int i;
        double sum = 0;
        for(i=0; i<100; i++){
            sum += evaluate();
        }
        sum /= 100;
        if(sum > switchScore){
            certainMode = true;
        }
        return sum;
    }
};

int main()
{
	srand((unsigned)time(NULL));
	Trainer t;
	//cout<<t.evaluate();
	/*
	Environment e;
	e.initialize();
	cout<<e.value<<'\n';
	t.trainActions(&e);
	*/
	
	int i,j;
	double val;
	for(j=0; j<50; j++){
	    t.trainGame();
	    val = t.multEval();
	    cout<<j<<' '<<val<<'\n';
	}
	
    return 0;
}








