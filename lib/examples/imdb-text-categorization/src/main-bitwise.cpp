#undef NDEBUG // I want assert to work

#include "tsetlini.hpp"
#include "basic_bit_vector_companion.hpp"

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
#include <chrono>


using aligned_vector_char = Tsetlini::aligned_vector_char;
using label_vector_type = Tsetlini::label_vector_type;
using bit_vector_uint64 = Tsetlini::bit_vector_uint64;

std::vector<std::string>
read_file(std::string const & fname)
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


std::vector<aligned_vector_char> read_data_as_vec(std::string const & fname, int ncol)
{
    auto const lines = read_file(fname);

    std::vector<aligned_vector_char> rv;

    std::transform(lines.cbegin(), lines.cend(), std::back_inserter(rv),
        [ncol](std::string const & s)
        {
            std::istringstream  ss(s);
            aligned_vector_char rv;

            std::generate_n(std::back_inserter(rv), ncol,
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


auto to_bitvector = [](std::vector<aligned_vector_char> const & X)
{
    std::vector<Tsetlini::bit_vector_uint64> rv;
    rv.reserve(X.size());

    std::transform(X.cbegin(), X.cend(), std::back_inserter(rv),
        [](auto const & sample){ return basic_bit_vectors::from_range<std::uint64_t>(sample.cbegin(), sample.cend()); }
    );

    return rv;
};


auto split_Xy = [](std::vector<aligned_vector_char> const & v)
{
    std::vector<aligned_vector_char> X;
    label_vector_type y;

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
    auto constexpr NCOL = 5000 + 1;

    auto const train_Xy = read_data_as_vec("IMDBTrainingData.txt", NCOL);
    auto const test_Xy = read_data_as_vec("IMDBTestData.txt", NCOL);

    if (train_Xy.size() == 0)
    {
        std::cout << R"(
Could not read from file IMDBTrainingData.txt. It either does not exist
or is not readable.
Please run produce_dataset.py script and move created .txt files to the folder with the executable
)";
        return 1;
    }
    if (test_Xy.size() == 0)
    {
        std::cout << R"(
Could not read from file IMDBTestData.txt. It either does not exist
or is not readable.
Please run produce_dataset.py script and move created .txt files to the folder with the executable
)";
        return 1;
    }


    auto const [itrain_X, train_y] = split_Xy(train_Xy);
    auto const [itest_X, test_y] = split_Xy(test_Xy);

    // binary-encoded data
    auto const train_X = to_bitvector(itrain_X);
    auto const test_X = to_bitvector(itest_X);

    constexpr auto SAMPLE_SZ = NCOL - 1u;
    assert(train_X.front().size() == SAMPLE_SZ);
    assert(test_X.front().size() == SAMPLE_SZ);


    auto error_printer = [](Tsetlini::status_message_t && msg)
    {
        std::cout << msg.second << '\n';
        return msg;
    };

    auto & now = std::chrono::high_resolution_clock::now;
    auto as_ms = [](auto && tp)
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(tp).count() / 1000.f;
    };

    Tsetlini::make_classifier_bitwise(R"({
            "threshold": 80,
            "s": 27.0,
            "number_of_pos_neg_clauses_per_label": 20000,
            "number_of_states": 127,
            "boost_true_positive_feedback": 1,
            "random_state": 1,
            "clause_output_tile_size": 64,
            "n_jobs": 2,
            "verbose": false
        })")
        .leftMap(error_printer)
        .rightMap([&](Tsetlini::ClassifierBitwise && clf)
        {
            std::chrono::high_resolution_clock::time_point time0{};

            for (auto epoch = 1u; epoch <= 50; ++epoch)
            {
                std::cout << "EPOCH " << epoch << '\n';

                time0 = now();

                auto status = clf.partial_fit(train_X, train_y, 2, 1);
                printf("Training Time: %.1f s\n", as_ms(now() - time0));

                time0 = now();

                clf.evaluate(test_X, test_y)
                    .leftMap(error_printer)
                    .rightMap([](auto acc){ printf("Test Accuracy: %.2f\n", acc * 100); return acc; });

                printf("Evaluation Time: %.1f s\n\n", as_ms(now() - time0) / 2.);
            }

            return clf;
        });

    return 0;
}
