#undef NDEBUG // I want assert to work

#include "tsetlini.hpp"
#include "tsetlini_strong_params.hpp"
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
#include <memory>
#include <cstring>
#include <cstdio>


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


auto to_bitvector = [](std::vector<aligned_vector_char> const & X)
{
    std::vector<Tsetlini::bit_vector_uint64> rv;
    rv.reserve(X.size());

    std::transform(X.cbegin(), X.cend(), std::back_inserter(rv),
        [](auto const & sample){ return basic_bit_vectors::from_range<std::uint64_t>(sample.cbegin(), sample.cend()); }
    );

    return rv;
};


int parse_args(int argc, char* argv[], unsigned int & nepochs)
{
    bool show_help = false;
    int c = 0;

    while (--argc > 0 && (*++argv)[0] == '-')
    {
        while ((c = *++argv[0]))
        {
            switch (c)
            {
                case 'n':
                {
                    if (--argc > 0)
                    {
                        auto parsed = atoi(argv[1]);
                        if (parsed >= 1)
                        {
                            nepochs = parsed;
                        }
                        else
                        {
                            fprintf(stderr, "Invalid nepochs value passed: %u\n", parsed);
                            argc = 0;
                        }

                        argv++;
                        *argv+= strlen(*argv) - 1;
                    }
                    break;
                }
                case 'h':
                    show_help = true;
                    break;

                default:
                {
                    fprintf(stderr, "Illegal option [%c]\n", (char)c);
                    argc = 0;
                    break;
                }
            }
        }
    }

    if (show_help or (argc < 0))
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "Usage: mnist-digits-2-new-bitwise-weighted [options]\n\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "         -n UINT   number of epochs to train, >= 1 [%u]\n", nepochs);
        fprintf(stderr, "         -h        show help\n");

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int main(int argc, char **argv)
{
    auto constexpr NCOL = 28 * 28 + 1;
    unsigned int nepochs = 60;

    if (parse_args(argc, argv, nepochs) != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }

    auto const train_Xy = read_data_as_vec("MNISTTraining.txt", NCOL);
    auto const test_Xy = read_data_as_vec("MNISTTest.txt", NCOL);

    if (train_Xy.size() == 0)
    {
        std::cout << R"(
Could not read from file MNISTTraining.txt. It either does not exist
or is not readable.
Please download and unzip it from https://github.com/cair/fast-tsetlin-machine-with-mnist-demo
You can issue the following command:
$> curl --remote-name-all https://github.com/cair/fast-tsetlin-machine-with-mnist-demo/raw/6d317dddcdb610c23deb89018d570bfc1b225657/BinarizedMNISTData.zip
or
$> wget https://github.com/cair/fast-tsetlin-machine-with-mnist-demo/raw/6d317dddcdb610c23deb89018d570bfc1b225657/BinarizedMNISTData.zip
)";
        return 1;
    }
    if (test_Xy.size() == 0)
    {
        std::cout << R"(
Could not read from file MNISTTest.txt. It either does not exist
or is not readable.
Please download and unzip it from https://github.com/cair/fast-tsetlin-machine-with-mnist-demo
You can issue the following command:
$> curl --remote-name-all https://github.com/cair/fast-tsetlin-machine-with-mnist-demo/raw/6d317dddcdb610c23deb89018d570bfc1b225657/BinarizedMNISTData.zip
or
$> wget https://github.com/cair/fast-tsetlin-machine-with-mnist-demo/raw/6d317dddcdb610c23deb89018d570bfc1b225657/BinarizedMNISTData.zip
)";
        return 1;
    }

    // integer-encoded data
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
        return std::move(msg);
    };

    auto & now = std::chrono::high_resolution_clock::now;
    auto as_ms = [](auto && tp)
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(tp).count() / 1000.f;
    };

    Tsetlini::make_classifier_bitwise(
        Tsetlini::threshold_t{5000},
        Tsetlini::specificity_t{10.0},
        Tsetlini::number_of_physical_classifier_clauses_per_label_t{4000},
        Tsetlini::number_of_states_t{127},
        Tsetlini::boost_tpf_t{true},
        Tsetlini::random_seed_t{1},
        Tsetlini::weighted_flag_t{true},
        Tsetlini::clause_output_tile_size_t{64},
        Tsetlini::number_of_jobs_t{2},
        Tsetlini::verbosity_t{false}
        )
        .leftMap(error_printer)
        .rightMap([&, train_y = train_y, test_y = test_y](Tsetlini::ClassifierBitwise && clf)
        {
            std::chrono::high_resolution_clock::time_point time0{};

            for (auto epoch = 1u; epoch <= nepochs; ++epoch)
            {
                std::cout << "EPOCH " << epoch << '\n';

                time0 = now();

                auto status = clf.partial_fit(train_X, train_y, Tsetlini::max_number_of_labels_t{10}, Tsetlini::number_of_epochs_t{1});
                printf("Training Time: %.1f s\n", as_ms(now() - time0));

                time0 = now();

                clf.evaluate(test_X, test_y)
                    .leftMap(error_printer)
                    .rightMap([](auto acc){ printf("Test Accuracy: %.2f\n", acc * 100); return acc; });

                printf("Evaluation Time: %.1f s\n\n", as_ms(now() - time0) / 2.);
            }

            return std::move(clf);
        });

    return EXIT_SUCCESS;
}
