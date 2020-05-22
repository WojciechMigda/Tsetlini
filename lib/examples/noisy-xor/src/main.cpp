#undef NDEBUG // I want assert to work

#include "tsetlini.hpp"

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


using aligned_vector_char = Tsetlini::aligned_vector_char;
using aligned_vector_int = Tsetlini::aligned_vector_int;

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


    auto const [train_X, train_y] = split_Xy(train_Xy);
    auto const [test_X, test_y] = split_Xy(test_Xy);

    assert(train_X.front().size() == 12);
    assert(test_X.front().size() == 12);

    auto error_printer = [](Tsetlini::status_message_t && msg)
    {
        std::cout << msg.second << '\n';
        return msg;
    };


    Tsetlini::make_classifier(R"({
            "threshold": 15,
            "s": 3.9,
            "number_of_pos_neg_clauses_per_label": 5,
            "number_of_states": 100,
            "boost_true_positive_feedback": 0,
            "random_state": 1,
            "counting_type": "int32",
            "n_jobs": 1,
            "verbose": false
        })")
        .leftMap(error_printer)
        .rightMap([&](Tsetlini::Classifier && clf)
        {
            // Training of the Tsetlin Machine in batch mode. The Tsetlin Machine can also be trained online
            auto status = clf.fit(train_X, train_y, 2, 200);

            // Some performance statistics
            clf.evaluate(test_X, test_y)
                .leftMap(error_printer)
                .rightMap([](auto acc){ std::cout << "Accuracy on test data (no noise): " << acc << '\n'; return acc; });

            clf.evaluate(train_X, train_y)
                .leftMap(error_printer)
                .rightMap([](auto acc){ std::cout << "Accuracy on training data (40% noise): " << acc << "\n\n"; return acc; });

            clf.predict(aligned_vector_char{1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0})
                .rightMap([](auto label){ std::cout << "Prediction: x1 = 1, x2 = 0, ... -> y = " << label << '\n'; return label; })
                .leftMap(error_printer);

            clf.predict(aligned_vector_char{0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0})
                .rightMap([](auto label){ std::cout << "Prediction: x1 = 0, x2 = 1, ... -> y = " << label << '\n'; return label; })
                .leftMap(error_printer);

            clf.predict(aligned_vector_char{0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0})
                .rightMap([](auto label){ std::cout << "Prediction: x1 = 0, x2 = 0, ... -> y = " << label << '\n'; return label; })
                .leftMap(error_printer);

            clf.predict(aligned_vector_char{1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0})
                .rightMap([](auto label){ std::cout << "Prediction: x1 = 1, x2 = 1, ... -> y = " << label << '\n'; return label; })
                .leftMap(error_printer);

            return clf;
        });

    return 0;
}
