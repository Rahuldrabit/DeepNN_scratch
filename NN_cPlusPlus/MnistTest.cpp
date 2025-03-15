#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

using namespace std;

// Function to read MNIST images
vector<vector<double>> read_mnist_images(const string &path) {
    ifstream file(path, ios::binary);
    if (!file) {
        cerr << "Cannot open file: " << path << endl;
        return {};
    }

    auto read_uint32 = [](ifstream &f) {
        uint32_t val;
        f.read(reinterpret_cast<char*>(&val), 4);
        return __builtin_bswap32(val);
    };

    uint32_t magic = read_uint32(file);
    if (magic != 2051) {
        cerr << "Invalid MNIST image file!" << endl;
        return {};
    }

    uint32_t num_images = read_uint32(file);
    uint32_t rows = read_uint32(file);
    uint32_t cols = read_uint32(file);

    vector<vector<double>> images;
    for (size_t i = 0; i < num_images; ++i) {
        vector<double> image(rows * cols);
        for (size_t j = 0; j < rows * cols; ++j) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            image[j] = pixel / 255.0;
        }
        images.push_back(image);
    }

    return images;
}

// Function to read MNIST labels
vector<int> read_mnist_labels(const string &path) {
    ifstream file(path, ios::binary);
    if (!file) {
        cerr << "Cannot open file: " << path << endl;
        return {};
    }

    auto read_uint32 = [](ifstream &f) {
        uint32_t val;
        f.read(reinterpret_cast<char*>(&val), 4);
        return __builtin_bswap32(val);
    };

    uint32_t magic = read_uint32(file);
    if (magic != 2049) {
        cerr << "Invalid MNIST label file!" << endl;
        return {};
    }

    uint32_t num_labels = read_uint32(file);

    vector<int> labels;
    for (size_t i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels.push_back(label);
    }

    return labels;
}

// Activation functions
double relu(double x) { return max(0.0, x); }
vector<double> softmax(const vector<double> &x) {
    vector<double> res(x.size());
    double max_x = *max_element(x.begin(), x.end());
    double sum = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
        res[i] = exp(x[i] - max_x);
        sum += res[i];
    }

    for (auto &val : res)
        val /= sum;
    return res;
}

class NeuralNetwork {
    int input_size, hidden_size, output_size;
    vector<vector<double>> W1, W2;
    vector<double> b1, b2;

public:
    NeuralNetwork(int input, int hidden, int output)
        : input_size(input), hidden_size(hidden), output_size(output),
          W1(input, vector<double>(hidden)),
          W2(hidden, vector<double>(output)),
          b1(hidden), b2(output) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-0.5, 0.5);

        for (int i = 0; i < input; ++i)
            for (int j = 0; j < hidden; ++j)
                W1[i][j] = dis(gen);

        for (int j = 0; j < hidden; ++j)
            b1[j] = dis(gen);

        for (int i = 0; i < hidden; ++i)
            for (int j = 0; j < output; ++j)
                W2[i][j] = dis(gen);

        for (int j = 0; j < output; ++j)
            b2[j] = dis(gen);
    }

    vector<double> forward(const vector<double> &input) {
        vector<double> hidden(hidden_size);
        for (int j = 0; j < hidden_size; ++j) {
            hidden[j] = b1[j];
            for (int i = 0; i < input_size; ++i)
                hidden[j] += input[i] * W1[i][j];
            hidden[j] = relu(hidden[j]);
        }

        vector<double> output(output_size);
        for (int j = 0; j < output_size; ++j) {
            output[j] = b2[j];
            for (int i = 0; i < hidden_size; ++i)
                output[j] += hidden[i] * W2[i][j];
        }

        return softmax(output);
    }

    void train(const vector<vector<double>> &inputs, const vector<int> &labels,
               int epochs, double lr) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0;
            int correct = 0;

            for (size_t i = 0; i < inputs.size(); ++i) {
                const auto &input = inputs[i];
                int label = labels[i];

                // Forward pass
                vector<double> hidden(hidden_size);
                for (int j = 0; j < hidden_size; ++j) {
                    hidden[j] = b1[j];
                    for (int k = 0; k < input_size; ++k)
                        hidden[j] += input[k] * W1[k][j];
                    hidden[j] = relu(hidden[j]);
                }

                vector<double> output(output_size);
                for (int j = 0; j < output_size; ++j) {
                    output[j] = b2[j];
                    for (int k = 0; k < hidden_size; ++k)
                        output[j] += hidden[k] * W2[k][j];
                }
                output = softmax(output);

                // Calculate loss
                double loss = -log(output[label] + 1e-10);
                total_loss += loss;

                // Calculate accuracy
                int pred = distance(output.begin(), max_element(output.begin(), output.end()));
                if (pred == label) correct++;

                // Backward pass
                vector<double> output_err(output_size);
                for (int j = 0; j < output_size; ++j)
                    output_err[j] = output[j] - (j == label);

                vector<double> hidden_err(hidden_size);
                for (int j = 0; j < hidden_size; ++j) {
                    double error = 0;
                    for (int k = 0; k < output_size; ++k)
                        error += output_err[k] * W2[j][k];
                    hidden_err[j] = error * (hidden[j] > 0 ? 1.0 : 0.0);
                }

                // Update weights and biases
                for (int j = 0; j < output_size; ++j) {
                    b2[j] -= lr * output_err[j];
                    for (int k = 0; k < hidden_size; ++k)
                        W2[k][j] -= lr * output_err[j] * hidden[k];
                }

                for (int j = 0; j < hidden_size; ++j) {
                    b1[j] -= lr * hidden_err[j];
                    for (int k = 0; k < input_size; ++k)
                        W1[k][j] -= lr * hidden_err[j] * input[k];
                }
            }

            cout << "Epoch " << epoch + 1 << "/" << epochs
                 << " - Loss: " << total_loss / inputs.size()
                 << " - Acc: " << 100.0 * correct / inputs.size() << "%" << endl;
        }
    }

    int predict(const vector<double> &input) {
        auto output = forward(input);
        return distance(output.begin(), max_element(output.begin(), output.end()));
    }
};

int main() {
    // Load MNIST data (update paths as needed)
    auto train_images = read_mnist_images("train-images-idx3-ubyte");
    auto train_labels = read_mnist_labels("train-labels-idx1-ubyte");
    auto test_images = read_mnist_images("t10k-images-idx3-ubyte");
    auto test_labels = read_mnist_labels("t10k-labels-idx1-ubyte");

    if (train_images.empty() || test_images.empty()) {
        cerr << "Error loading data!" << endl;
        return 1;
    }

    // Create and train network
    NeuralNetwork nn(784, 128, 10);
    nn.train(train_images, train_labels, 5, 0.01);

    // Test network
    int correct = 0;
    for (size_t i = 0; i < test_images.size(); ++i) {
        int pred = nn.predict(test_images[i]);
        if (pred == test_labels[i]) correct++;
    }

    cout << "Test Accuracy: " << 100.0 * correct / test_images.size() << "%" << endl;

    return 0;
}