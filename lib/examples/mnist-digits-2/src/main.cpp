#undef NDEBUG // I want assert to work

#include "tsetlini.hpp"
#include "tsetlini_strong_params.hpp"

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
#include <memory>


using aligned_vector_char = Tsetlini::aligned_vector_char;
using label_vector_type = Tsetlini::label_vector_type;

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
    auto constexpr NCOL = 28 * 28 + 1;

    auto const train_Xy = read_data_as_vec("MNISTTraining.txt", NCOL);
    auto const test1_Xy = read_data_as_vec("MNISTTest.txt", NCOL);
    auto const test2_Xy = read_data_as_vec("MNISTTest2.txt", NCOL);

    if (train_Xy.size() == 0)
    {
        std::cout << R"(
Could not read from file MNISTTraining.txt. It either does not exist
or is not readable.
Please download and unzip it from https://github.com/cair/fast-tsetlin-machine-with-mnist-demo
You can issue the following command:
$> curl --remote-name-all https://github.com/cair/fast-tsetlin-machine-with-mnist-demo/raw/ca5ae464886d75da4247e7108ed4d17ea08845b7/BinarizedMNISTData.zip
or
$> wget https://github.com/cair/fast-tsetlin-machine-with-mnist-demo/raw/ca5ae464886d75da4247e7108ed4d17ea08845b7/BinarizedMNISTData.zip
)";
        return 1;
    }
    if (test1_Xy.size() == 0)
    {
        std::cout << R"(
Could not read from file MNISTTest.txt. It either does not exist
or is not readable.
Please download and unzip it from https://github.com/cair/fast-tsetlin-machine-with-mnist-demo
You can issue the following command:
$> curl --remote-name-all https://github.com/cair/fast-tsetlin-machine-with-mnist-demo/raw/ca5ae464886d75da4247e7108ed4d17ea08845b7/BinarizedMNISTData.zip
or
$> wget https://github.com/cair/fast-tsetlin-machine-with-mnist-demo/raw/ca5ae464886d75da4247e7108ed4d17ea08845b7/BinarizedMNISTData.zip
)";
        return 1;
    }
    if (test2_Xy.size() == 0)
    {
        std::cout << R"(
Could not read from file MNISTTest.txt. It either does not exist
or is not readable.
Please download and unzip it from https://github.com/cair/fast-tsetlin-machine-with-mnist-demo
You can issue the following command:
$> curl --remote-name-all https://github.com/cair/fast-tsetlin-machine-with-mnist-demo/raw/ca5ae464886d75da4247e7108ed4d17ea08845b7/BinarizedMNISTData.zip
or
$> wget https://github.com/cair/fast-tsetlin-machine-with-mnist-demo/raw/ca5ae464886d75da4247e7108ed4d17ea08845b7/BinarizedMNISTData.zip
)";
        return 1;
    }


    auto const [train_X, train_y] = split_Xy(train_Xy);
    auto const [test1_X, test1_y] = split_Xy(test1_Xy);
    auto const [test2_X, test2_y] = split_Xy(test2_Xy);

    constexpr auto SAMPLE_SZ = NCOL - 1u;
    assert(train_X.front().size() == SAMPLE_SZ);
    assert(test1_X.front().size() == SAMPLE_SZ);
    assert(test2_X.front().size() == SAMPLE_SZ);


    auto error_printer = [](Tsetlini::status_message_t && msg)
    {
        std::cout << msg.second << '\n';
        return std::move(msg);
    };

    auto & now = std::chrono::high_resolution_clock::now;
    auto as_ms = [](auto && tp)
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(tp).count() / 1000.f;
    };

    Tsetlini::make_classifier_classic(
        Tsetlini::threshold_t{25},
        Tsetlini::specificity_t{10.0},
        Tsetlini::number_of_physical_classifier_clauses_per_label_t{1000},
        Tsetlini::number_of_states_t{127},
        Tsetlini::boost_tpf_t{true},
        Tsetlini::random_seed_t{1},
        Tsetlini::clause_output_tile_size_t{64},
        Tsetlini::number_of_jobs_t{2},
        Tsetlini::verbosity_t{false}
        )
        .leftMap(error_printer)
        .rightMap([&, train_X = train_X, train_y = train_y,
                   test1_X = test1_X, test1_y = test1_y,
                   test2_X = test2_X, test2_y = test2_y](Tsetlini::ClassifierClassic && clf)
        {
            std::chrono::high_resolution_clock::time_point time0{};

            for (auto epoch = 1u; epoch <= 250; ++epoch)
            {
                std::cout << "EPOCH " << epoch << '\n';

                time0 = now();

                auto status = clf.partial_fit(train_X, train_y, Tsetlini::max_number_of_labels_t{10}, Tsetlini::number_of_epochs_t{1});
                printf("Training Time: %.1f s\n", as_ms(now() - time0));

                time0 = now();

                clf.evaluate(test1_X, test1_y)
                    .leftMap(error_printer)
                    .rightMap([](auto acc){ printf("Accuracy Dataset I: %.2f\n", acc * 100); return acc; });

                clf.evaluate(test2_X, test2_y)
                    .leftMap(error_printer)
                    .rightMap([](auto acc){ printf("Accuracy Dataset II: %.2f\n", acc * 100); return acc; });

                printf("Evaluation Time: %.1f s\n\n", as_ms(now() - time0) / 2.);
            }

            return std::move(clf);
        });

    return EXIT_SUCCESS;
}
