#include<iostream>
#include<cstdlib>
#include<vector>
#include<assert.h>
#include<math.h>
#include <fstream>
#include <sstream>

using namespace std;

// Silly class to read training data from a text file -- Replace This.
// Replace class TrainingData with whatever you need to get input data into the
// program, e.g., connect to a database, or take a stream of data from stdin, or
// from a file specified by a command line argument, etc.

class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}
struct Connection{
    double weight ; 
    double delWeights ; 
} ;

class Neuron ; 

typedef vector<Neuron> Layer ; 


class Neuron {
    private : 
        static double eta ; 

        static double alpha ; 

        static double activationFunction(double sum){
            //tanh  
            return tanh(sum) ; 
        }

        static double activationFunctionDerivative(double sum){
            return 1.0 - sum*sum ; 

        }

        double sumDOW(const Layer &nextLayer){
            double sum = 0.0 ; 

            for(unsigned n =0 ;  n< nextLayer.size() ; ++ n) {
                sum += outputWeights[n].weight * nextLayer[n].gradient ; 
            }

            return sum ; 
        }
        static double randomWeight() {return rand() / double(RAND_MAX) ; }
        double outputVal ; 
        vector<Connection> outputWeights ; 
        unsigned myIndex ; 
        double gradient ; 

    public : 
        Neuron(unsigned numOfOutputs , unsigned myIndex);
         

        void setOutputval(double val){
            outputVal = val ; 
        }

        double getOutputVal(){
            return outputVal  ; 
        }
        void feedForward(const Layer &prevLayer)   ;

        void calcOutputGradients ( double targetVal){
            double delta = targetVal - outputVal ; 
            gradient = delta * activationFunctionDerivative(outputVal) ; 
        }

        void calcHiddenGradients (const Layer &nextLayer){
            double dow = sumDOW(nextLayer) ; 
            gradient = dow * activationFunctionDerivative(outputVal) ; 
        }

        void updateInputWeight(Layer &prevLayer){
            for(unsigned n =0 ; n< prevLayer.size() ; ++n){
                Neuron &neuron = prevLayer[n] ; 
                double oldDeltaWeights = neuron.outputWeights[myIndex].delWeights ; 

                double newDeltaWeights = 
                    eta 
                 * neuron.getOutputVal()
                 * gradient
                 + alpha 
                 *oldDeltaWeights; 

                 neuron.outputWeights[myIndex].delWeights = newDeltaWeights ; 
                 neuron.outputWeights[myIndex].weight += newDeltaWeights ; 
            }
        }
    
};

double Neuron::eta = .20 ;  // learning rate 
double Neuron::alpha = .5;  // momentum 


void Neuron::feedForward(const Layer &prevLayer){
            double sum = 0.0 ; 

        for(unsigned n =0 ; n< prevLayer.size() ; ++n){
            sum += prevLayer[n].getOutputVal() * prevLayer[n].outputWeights[myIndex].weight ; 
        }

        outputVal = Neuron::activationFunction(sum) ;
}

Neuron::Neuron(unsigned numOfOutputs , unsigned MyIndex){
    for(int i =0 ; i<numOfOutputs ; i++){
        outputWeights.push_back(Connection() ) ; 
        outputWeights.back().weight = randomWeight() ; 
    }
    MyIndex = MyIndex ; 
}


class Net{
    public : 
        Net(const vector<unsigned> &topology) ; 
        void feedForward(const vector <double>  &inputVals) ; 
        void backProp(const vector<double>  &targetVals) ; 
        void getResults(vector <double>  &resultVals) const  ; 
          double getRecentAverageError(void) const { return recentAverageError; }

    private :
        vector<Layer> layers ; 
        double error ; 
        double recentAverageError ; 
        static double recentAverageSmoothingFactor;
};


double Net::recentAverageSmoothingFactor = 100.0;


void Net::feedForward(const vector<double> &inputVals){
    assert(inputVals.size() == layers[0].size() -1 ) ;

    for(int i = 0 ; i<inputVals.size() ; i++){
        layers[0][i].setOutputval(inputVals[i]) ;  
    }

    for(int ln = 1 ; ln < layers.size() ; ln++)
    {
        Layer &prevLayer = layers[ln-1] ; 
        for(int n = 0 ; n < layers[ln].size()-1 ; ln++){
            layers[ln][n].feedForward(prevLayer) ;
        }

    }
}

void Net::backProp(const vector<double>  &targetVals){
    // calculate RMS ;
    Layer &outputLayer = layers.back();  
    error = 0.0; 

    for(int i =0 ; i<outputLayer.size() -1 ; i++){
        double delta = targetVals[i] - outputLayer[i].getOutputVal() ;
        error += delta* delta ;  
    } 
    error = error / outputLayer.size() -1 ; 
    error = sqrt(error) ; // ------>>RMS<<---------

    recentAverageError =
            (recentAverageError * recentAverageSmoothingFactor + error)
            / (recentAverageSmoothingFactor + 1.0);
    //calculate outputLayer Gradient 

    for(int n = 0 ; n< outputLayer.size() -1 ; ++n){
        outputLayer[n].calcOutputGradients(targetVals[n]) ;
    }
    // calculate gradient of hiddenLayer 

    for(int layerNum = 0 ; layerNum < outputLayer.size() -2 ; ++layerNum){
        Layer &hiddenLayer = layers[layerNum]  ; 
        Layer &nextLayer = layers[layerNum+1] ; 

        for(int n =0 ; n < hiddenLayer.size() ; ++n){
            hiddenLayer[n].calcHiddenGradients(nextLayer) ; 
        }
        
    }
    // for all layers from outputs to first hidden layer update connection weights 

    for(int layerNum = 0 ; layerNum < outputLayer.size() -1 ; ++layerNum){
        Layer &currentLayer = layers[layerNum] ;
        Layer &prevLayer = layers[layerNum-1] ;  

        for(int n = 0 ; n< currentLayer.size() ; n++){
            currentLayer[n].updateInputWeights(prevLayer) ; 
        }
    }
}


void Net::getResults(vector<double> &resultVals) const {
    resultVals.clear() ; 

    for( unsigned n =0 ; n< layers.back().size() -1 ; ++n) {
        resultVals.push_back(layers.back()[n].getOutputVal()) ; 
    }
}

Net::Net(const vector<unsigned> &topology){
        int numLayers = topology.size() ; 

        for(int i = 0 ; i <= numLayers ; i++){
            layers.push_back(Layer()) ;
            unsigned numOfOutputs = i == topology.size() - 1 ? 0  : topology[i+1] ; 
            for( int j = 0 ; j <= topology[i] ; j++){
                layers.back().push_back(Neuron(numOfOutputs , j));
            }

            layers.back().back().setOutputVal(1.0) ; 
        }
    }

void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}


int main() {
    TrainingData trainData("/tmp/trainingData.txt");

    // e.g., { 3, 2, 1 }
    vector<unsigned> topology;
    trainData.getTopology(topology);

    Net myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEof()) {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != topology[0]) {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual output results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: "
                << myNet.getRecentAverageError() << endl;
    }

    cout << endl << "Done" << endl;
}