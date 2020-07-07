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
#include <algorithm>
#include <random>
#include <cmath>

using aligned_vector_char = Tsetlini::aligned_vector_char;
using aligned_vector_int = Tsetlini::aligned_vector_int;

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


auto split_Xy = [](std::vector<aligned_vector_char> const & v)
{
    std::vector<aligned_vector_char> X;
    std::vector<char> y;

    for (auto const & row : v)
    {
        y.push_back(row.back());
        X.push_back(row);
        X.back().resize(row.size() - 1);
    }

    return std::make_tuple(X, y);
};


template<typename InputIt, typename RealType=float>
auto stdev_mean(InputIt first, InputIt last)
{
    auto const N = std::distance(first, last);
    RealType const sum = std::accumulate(first, last, 0.0);
    auto const mean = sum / N;

    std::vector<RealType> diff(N);
    std::transform(first, last, diff.begin(), [mean](RealType x) { return x - mean; });
    RealType const sq_sum = std::inner_product(diff.cbegin(), diff.cend(), diff.cbegin(), 0.0);
    RealType const stdev = std::sqrt(sq_sum / N);

    return std::make_pair(mean, stdev);
}


int main()
{
    auto constexpr NCOLS = 300 + 1;

    // Loading of training and test data
    auto const df_Xy = read_data_as_vec("BreastCancerData.txt", NCOLS);
    if (df_Xy.size() == 0)
    {
        std::cout << R"(
Could not read from file BreastCancerData.txt. It either does not exist
or is not readable.
Please run produce_dataset.py script and move created .txt files to the folder with the executable
)";
        return 1;
    }


    auto const [df_X, df_y] = split_Xy(df_Xy);

    assert(df_X.front().size() == NCOLS - 1);


    constexpr auto ensemble_size = 100u;
    std::vector<Tsetlini::real_type> accuracy_train(ensemble_size);
    std::vector<Tsetlini::real_type> accuracy_test(ensemble_size);

    std::vector<int> ix(df_X.size());
    std::iota(ix.begin(), ix.end(), 0);
    auto const seed = std::random_device{}();
    std::mt19937 gen(seed);

    std::size_t const PIVOT = df_X.size() * 0.8;
    std::vector<aligned_vector_char> train_X(PIVOT);
    std::vector<int> train_y(PIVOT);
    std::vector<aligned_vector_char> test_X(df_X.size() - PIVOT);
    std::vector<int> test_y(df_y.size() - PIVOT);

    auto error_printer = [](Tsetlini::status_message_t const & msg)
    {
        std::cout << msg.second << '\n';
        return msg;
    };


    for (auto it = 0u; it < ensemble_size; ++it)
    {
        std::cout << "ENSEMBLE " << it + 1 << '\n';

        std::shuffle(ix.begin(), ix.end(), std::mt19937(gen));

        for (auto rit = 0u; rit < PIVOT; ++rit)
        {
            train_X[rit] = df_X[ix[rit]];
            train_y[rit] = df_y[ix[rit]];
        }
        for (auto rit = PIVOT; rit < df_X.size(); ++rit)
        {
            test_X[rit - PIVOT] = df_X[ix[rit]];
            test_y[rit - PIVOT] = df_y[ix[rit]];
        }

        auto clf = Tsetlini::make_classifier_classic(R"({
            "threshold": 40,
            "s": 5.0,
            "number_of_pos_neg_clauses_per_label": 1600,
            "number_of_states": 127,
            "boost_true_positive_feedback": 1,
            "random_state": 1,
            "n_jobs": 1,
            "verbose": false
        })").leftMap(error_printer)
            .rightMap([&](auto && clf)
            {
                auto status = clf.fit(train_X, train_y, 2, 25);

                clf.evaluate(test_X, test_y)
                    .leftMap(error_printer)
                    .rightMap([&](auto acc)
                    {
                        accuracy_test[it] = 100. * acc;

                        auto [mean, stdev] = stdev_mean(accuracy_test.data(), accuracy_test.data() + it + 1);
                        printf("Average accuracy on test data: %.2f +/- %.2f\n", mean, 1.96 * stdev / std::sqrt(it + 1));

                        return acc;
                    });

                return clf;
            })
#if 0
            .rightMap([&](auto && clf)
            {
                clf.evaluate(train_X, train_y)
                    .leftMap(error_printer)
                    .rightMap([&](auto acc)
                    {
                        accuracy_train[it] = 100. * acc;

                        auto [mean, stdev] = stdev_mean(accuracy_train.data(), accuracy_train.data() + it + 1);
                        printf("Average accuracy on train data: %.2f +/- %.2f\n", mean, 1.96 * stdev / std::sqrt(it + 1));

                        return acc;
                    });

                return clf;
            })
#endif
            ;
    }

    return 0;
}
