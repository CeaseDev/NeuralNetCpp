#include <iostream>
#include <fstream>
#include <vector>

// Define the XNOR function
int xnor(int a, int b) {
    return (a && b) || (!a && !b);
}

int main() {
    std::ofstream data_file("xnor_training_data.txt");

    // Generate all possible input combinations (0,0), (0,1), (1,0), (1,1)
    std::vector<int> inputs = {0, 0, 0, 1, 1, 0, 1, 1};

    // Generate corresponding XNOR outputs
    std::vector<int> outputs;
    for (int i = 0; i < 8; i += 2) {
        int result = xnor(inputs[i], inputs[i + 1]);
        outputs.push_back(result);
    }

    // Write input-output pairs to the data file
    for (size_t i = 0; i < inputs.size(); i += 2) {
        data_file << inputs[i] << " " << inputs[i + 1] << " " << outputs[i / 2] << "\n";
    }

    data_file.close();
    std::cout << "XNOR training data generated and saved to 'xnor_training_data.txt'.\n";

    return 0;
}
