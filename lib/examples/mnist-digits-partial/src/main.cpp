#undef NDEBUG // I want assert to work

#include "tsetlini.hpp"
#include "tsetlini_private.hpp"
#include "tsetlini_state_json.hpp"

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
        y.push_back(row.front());
        X.emplace_back(std::next(row.cbegin()), row.cend());
    }

    return std::make_tuple(X, y);
};

#define PP_STR(a) XSTR(a)
#define XSTR(a) #a

int main()
{
    auto constexpr NCOL = 1 + 64 * 3;

    auto const train_Xy = read_data_as_vec("digits_1.txt", NCOL);
    auto const test_Xy = read_data_as_vec("digits_2.txt", NCOL);

    if (train_Xy.size() == 0)
    {
        std::cout << R"(
Could not read from file digits_1.txt. It either does not exist
or is not readable.
Please run make_data.py to generate this file and copy it to the current working folder.
)";
        return 1;
    }

    if (test_Xy.size() == 0)
    {
        std::cout << R"(
Could not read from file digits_2.txt. It either does not exist
or is not readable.
Please run make_data.py to generate this file and copy it to the current working folder.
)";
        return 1;
    }

    auto const [train_X, train_y] = split_Xy(train_Xy);
    auto const [test_X, test_y] = split_Xy(test_Xy);

    constexpr auto SAMPLE_SZ = NCOL - 1u;
    assert(train_X.front().size() == SAMPLE_SZ);
    assert(test_X.front().size() == SAMPLE_SZ);

    auto error_printer = [](Tsetlini::status_message_t && msg)
    {
        std::cout << msg.second << '\n';
        return msg;
    };

    auto const params = R"({
        "threshold": 10,
        "s": 3.0,
        "number_of_pos_neg_clauses_per_label": 50,
        "number_of_states": 100,
        "boost_true_positive_feedback": 0,
        "random_state": 1,
        "clause_output_tile_size": 64,
        "n_jobs": 2,
        "verbose": false
    })";

    puts("===[ 2 + 3 epoch fit ]=============================================");

    Tsetlini::make_classifier_classic(params)
        .leftMap(error_printer)
        .rightMap([&](Tsetlini::ClassifierClassic && clf)
        {
            auto const NEPOCHS = 2;
            auto status = clf.fit(train_X, train_y, 10, NEPOCHS);

            clf.evaluate(test_X, test_y)
                .leftMap(error_printer)
                .rightMap([&](auto acc)
                {
                    printf("Accuracy after %d epochs: %.2f\n", NEPOCHS, acc * 100);
                    return acc;
                }
            );

            return clf;
        })
        .rightMap([&](Tsetlini::ClassifierClassic && clf)
        {
            auto const state = clf.read_state();
            auto const j_state = Tsetlini::to_json_string(state);
            Tsetlini::ClassifierStateClassic new_state(clf.read_params());
            Tsetlini::from_json_string(new_state, j_state);

            Tsetlini::ClassifierClassic clf2(new_state);

            auto const NEPOCHS = 3u;
            auto status = clf2.partial_fit(train_X, train_y, 0, NEPOCHS);

            clf2.evaluate(test_X, test_y)
                .leftMap(error_printer)
                .rightMap([&](auto acc)
                {
                    printf("Accuracy after +%d epochs: %.2f\n", NEPOCHS, acc * 100);
                    return acc;
                }
            );

            return clf2;
        });

    Tsetlini::make_classifier_classic(params)
        .leftMap(error_printer)
        .rightMap([&](Tsetlini::ClassifierClassic && clf)
        {
            auto const NEPOCHS = 5;
            auto status = clf.fit(train_X, train_y, 10, NEPOCHS);

            clf.evaluate(test_X, test_y)
                .leftMap(error_printer)
                .rightMap([&](auto acc)
                {
                    printf("Accuracy after %d epochs: %.2f\n", NEPOCHS, acc * 100);
                    return acc;
                });

            return clf;
        });

    puts("===[ 49 + 1 epoch fit ]============================================");

    Tsetlini::make_classifier_classic(params)
        .leftMap(error_printer)
        .rightMap([&](Tsetlini::ClassifierClassic && clf)
        {
            auto const NEPOCHS = 49;
            auto status = clf.fit(train_X, train_y, 10, NEPOCHS);

            clf.evaluate(test_X, test_y)
                .leftMap(error_printer)
                .rightMap([&](auto acc)
                {
                    printf("Accuracy after %d epochs: %.2f\n", NEPOCHS, acc * 100);
                    return acc;
                }
            );

            return clf;
        })
        .rightMap([&](Tsetlini::ClassifierClassic && clf)
        {
            auto const state = clf.read_state();
            auto const j_state = Tsetlini::to_json_string(state);
            Tsetlini::ClassifierStateClassic new_state(clf.read_params());
            Tsetlini::from_json_string(new_state, j_state);

            Tsetlini::ClassifierClassic clf2(new_state);

            auto const NEPOCHS = 1u;
            auto status = clf2.partial_fit(train_X, train_y, 0, NEPOCHS);

            clf2.evaluate(test_X, test_y)
                .leftMap(error_printer)
                .rightMap([&](auto acc)
                {
                    printf("Accuracy after +%d epochs: %.2f\n", NEPOCHS, acc * 100);
                    return acc;
                }
            );

            return clf2;
        });

    Tsetlini::make_classifier_classic(params)
        .leftMap(error_printer)
        .rightMap([&](Tsetlini::ClassifierClassic && clf)
        {
            auto const NEPOCHS = 50;
            auto status = clf.fit(train_X, train_y, 10, NEPOCHS);

            clf.evaluate(test_X, test_y)
                .leftMap(error_printer)
                .rightMap([&](auto acc)
                {
                    printf("Accuracy after %d epochs: %.2f\n", NEPOCHS, acc * 100);
                    return acc;
                });

            return clf;
        });


    return 0;
}
