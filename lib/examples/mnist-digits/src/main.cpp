#undef NDEBUG // I want assert to work

#ifndef DATA_DIR
#error "DATA_DIR is not defined"
#endif

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


std::vector<aligned_vector_char> read_data_as_vec(std::string const & fname)
{
    auto const lines = read_file(fname);

    std::vector<aligned_vector_char> rv;

    std::transform(lines.cbegin(), lines.cend(), std::back_inserter(rv),
        [](std::string const & s)
        {
            std::istringstream  ss(s);
            aligned_vector_char rv;

            std::generate_n(std::back_inserter(rv), 1 + 64 * 3,
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
        y.push_back(row.front());
        X.emplace_back(std::next(row.cbegin()), row.cend());
    }

    return std::make_tuple(X, y);
};

#define PP_STR(a) XSTR(a)
#define XSTR(a) #a

int main()
{
    constexpr auto NFOLDS = 5u;

    std::vector<std::vector<aligned_vector_char>> X_folds;
    std::vector<label_vector_type> y_folds;

    for (auto ix = 0u; ix < NFOLDS; ++ix)
    {
        std::string const fname = PP_STR(DATA_DIR) "/digits_" + std::to_string(ix + 1) + ".txt";
        auto const Xy = read_data_as_vec(fname);

        if (Xy.size() == 0)
        {
            std::cerr << "Could not read from file " << fname << R"( . It
either is not readable or does not exist.)";
        }

        auto [X, y] = split_Xy(Xy);

        assert(X.front().size() == 3 * 64);

        X_folds.emplace_back(std::move(X));
        y_folds.emplace_back(std::move(y));
    }


    auto error_printer = [](Tsetlini::status_message_t && msg)
    {
        std::cout << msg.second << '\n';
        return std::move(msg);
    };


    std::vector<Tsetlini::real_type> accuracies(NFOLDS);

    for (auto it = 0u; it < NFOLDS; ++it)
    {
        std::vector<aligned_vector_char> X_train;
        label_vector_type y_train;

        for (auto fit = 0u; fit < NFOLDS; ++fit)
        {
            if (fit == it)
            {
                continue;
            }

            X_train.insert(X_train.end(), X_folds[fit].cbegin(), X_folds[fit].cend());
            y_train.insert(y_train.end(), y_folds[fit].cbegin(), y_folds[fit].cend());
        }

        Tsetlini::make_classifier_classic(
            Tsetlini::threshold_t{10},
            Tsetlini::specificity_t{3.0},
            Tsetlini::number_of_physical_classifier_clauses_per_label_t{100},
            Tsetlini::number_of_states_t{1000},
            Tsetlini::boost_tpf_t{false},
            Tsetlini::random_seed_t{1},
            Tsetlini::clause_output_tile_size_t{64},
            Tsetlini::number_of_jobs_t{2},
            Tsetlini::verbosity_t{false}
            )
            .leftMap(error_printer)
            .rightMap([&](Tsetlini::ClassifierClassic && clf)
            {
                auto status = clf.fit(X_train, y_train, Tsetlini::max_number_of_labels_t{10}, Tsetlini::number_of_epochs_t{300});

                clf.evaluate(X_folds[it], y_folds[it])
                    .leftMap(error_printer)
                    .rightMap([&](auto acc)
                    {
                        accuracies[it] = acc;

                        std::cout << "Fold " << it + 1 << ", accuracy: " << acc << '\n';

                        return acc;
                    });

                return std::move(clf);
            });
    }

    std::cout << "Mean accuracy: " << std::accumulate(accuracies.cbegin(), accuracies.cend(), 0.) / NFOLDS << '\n';

    return EXIT_SUCCESS;
}
