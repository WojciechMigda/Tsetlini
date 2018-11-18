#undef NDEBUG // I wan't assert to work

#include "tsetlin.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <tuple>
#include <cassert>
#include <iostream>
#include <cstdlib>


using aligned_vector_char = Tsetlin::aligned_vector_char;
using aligned_vector_int = Tsetlin::aligned_vector_int;

std::vector<std::string>
read_file(std::string && fname)
{
    std::ifstream fcsv(fname);
    std::vector<std::string> vcsv;

    for (std::string line; std::getline(fcsv, line); /* nop */)
    {
        vcsv.push_back(line);
    }
    fcsv.close();

    return vcsv;
}


std::vector<aligned_vector_char> read_data_as_vec(std::string && fname)
{
    auto const lines = read_file(std::move(fname));

    std::vector<aligned_vector_char>  rv;

    std::transform(lines.cbegin(), lines.cend(), std::back_inserter(rv),
        [](std::string const & s)
        {
            std::istringstream  ss(s);
            aligned_vector_char rv;

            std::generate_n(std::back_inserter(rv), 13,
                [&ss]()
                {
                    return *(std::istream_iterator<int>(ss));
                }
            );

            return rv;
        }
    );

    return rv;
}


auto split_Xy = [](std::vector<aligned_vector_char> const & v)
{
    std::vector<aligned_vector_char> X;
    std::vector<int> y;

    for (auto const & row : v)
    {
        y.push_back(row.back());
        X.push_back(row);
        X.back().resize(row.size() - 1);
    }

    return std::make_tuple(X, y);
};


int main()
{
    // Loading of training and test data
    auto const train_Xy = read_data_as_vec("NoisyXORTrainingData.txt");
    auto const test_Xy = read_data_as_vec("NoisyXORTestData.txt");

    if (train_Xy.size() == 0)
    {
        std::cout << R"(
Could not read from file NoisyXORTrainingData.txt. It either does not exist
or is not readable.
Please download it from https://github.com/cair/TsetlinMachineCython
You can issue the following command:
$> curl https://raw.githubusercontent.com/cair/TsetlinMachineCython/79f0be5c9b259d2364b4ec86d46bb6f9fd4ce787/NoisyXORTrainingData.txt
or
$> wget https://raw.githubusercontent.com/cair/TsetlinMachineCython/79f0be5c9b259d2364b4ec86d46bb6f9fd4ce787/NoisyXORTrainingData.txt
)";
        return 1;
    }
    if (test_Xy.size() == 0)
    {
        std::cout << R"(
Could not read from file NoisyXORTrainingData.txt. It either does not exist
or is not readable.
Please download it from https://github.com/cair/TsetlinMachineCython
You can issue the following command:
$> curl https://raw.githubusercontent.com/cair/TsetlinMachineCython/79f0be5c9b259d2364b4ec86d46bb6f9fd4ce787/NoisyXORTestData.txt
or
$> wget https://raw.githubusercontent.com/cair/TsetlinMachineCython/79f0be5c9b259d2364b4ec86d46bb6f9fd4ce787/NoisyXORTestData.txt
)";
        return 1;
    }


    // for compiler with support of structured bindings
    auto const [train_X, train_y] = split_Xy(train_Xy);
    auto const [test_X, test_y] = split_Xy(test_Xy);

    assert(train_X.front().size() == 12);
    assert(test_X.front().size() == 12);

    auto patch = Tsetlin::config_patch_from_json(R"({
        "threshold": 15,
        "s": 3.9,
        "number_of_pos_neg_clauses_per_class": 5,
        "number_of_states": 100,
        "number_of_features": 12,
        "number_of_classes": 2,
        "boost_true_positive_feedback": 0,
        "seed": 1,
        "verbose": false
    })");
    assert(patch.size() != 0);

    Tsetlin::Classifier tsetlin_machine(Tsetlin::make_classifier_state(patch));

    // Training of the Tsetlin Machine in batch mode. The Tsetlin Machine can also be trained online
    tsetlin_machine.fit(train_X, train_y, train_y.size(), 200);

    // Some performacne statistics

    std::cout << "Accuracy on test data (no noise): " <<
        tsetlin_machine.evaluate(test_X, test_y, test_y.size()) << '\n';

    std::cout << "Accuracy on training data (40% noise): " <<
        tsetlin_machine.evaluate(train_X, train_y, train_y.size()) << "\n\n";

    std::cout << "Prediction: x1 = 1, x2 = 0, ... -> y = " <<
        tsetlin_machine.predict(aligned_vector_char{1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0}) << '\n';
    std::cout << "Prediction: x1 = 0, x2 = 1, ... -> y = " <<
        tsetlin_machine.predict(aligned_vector_char{0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0}) << '\n';
    std::cout << "Prediction: x1 = 0, x2 = 0, ... -> y = " <<
        tsetlin_machine.predict(aligned_vector_char{0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0}) << '\n';
    std::cout << "Prediction: x1 = 1, x2 = 1, ... -> y = " <<
        tsetlin_machine.predict(aligned_vector_char{1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0}) << '\n';

    return 0;
}
