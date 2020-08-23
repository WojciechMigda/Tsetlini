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


struct Scaler
{
    explicit Scaler(int T) : m_T(T), m_min(0), m_max(0) {}

    void fit(std::vector<float> const & y)
    {
        auto const [it_min, it_max] = std::minmax_element(y.cbegin(), y.cend());

        m_min = *it_min;
        m_max = *it_max;
    }

    std::vector<int> transform(std::vector<float> const & y)
    {
        std::vector<int> rv(y.size());

        std::transform(y.cbegin(), y.cend(), rv.begin(),
            [this](float v)
            {
                return std::clamp<int>(std::round((v - m_min) / (m_max - m_min) * m_T), 0, m_T);
            });

        return rv;
    }

    std::vector<float> inverse_transform(std::vector<int> const & y)
    {
        std::vector<float> rv(y.size());

        std::transform(y.cbegin(), y.cend(), rv.begin(),
            [this](int v)
            {
                return v * (m_max - m_min) / m_T + m_min;
            });

        return rv;
    }

    int m_T;
    float m_min;
    float m_max;
};


float rms(std::vector<float> const & p, std::vector<float> const & q)
{
    float rv = 0;

    for (auto ix = 0u; ix < p.size(); ++ix)
    {
        auto const delta = p[ix] - q[ix];
        rv += delta * delta;
    }

    return std::sqrt(rv / p.size());
}


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
    auto constexpr NCOLS = 80;

    auto const df_X_int = read_data_as_vec("CaliforniaHousingData_X.txt", NCOLS);
    auto const df_y = [&df_X_int]()
        {
            auto const lines = read_file("CaliforniaHousingData_Y.txt");
            std::vector<float> rv(df_X_int.size());

            std::transform(lines.cbegin(), lines.cend(), rv.begin(), [](std::string const & ll){ return std::stof(ll); });

            return rv;
        }();

    if (df_X_int.size() == 0)
    {
        std::cout << R"(
Could not read from file CaliforniaHousingData_X.txt. It either does not exist
or is not readable.
Please run produce_dataset.py script and move created .txt files to the folder with the executable
)";
        return 1;
    }
    if (df_y.size() == 0)
    {
        std::cout << R"(
Could not read from file CaliforniaHousingData_Y.txt. It either does not exist
or is not readable.
Please run produce_dataset.py script and move created .txt files to the folder with the executable
)";
        return 1;
    }

    auto const df_X = to_bitvector(df_X_int);

    assert(df_X.front().size() == NCOLS);
    assert(df_X.size() == df_y.size());


    constexpr auto ensemble_size = 25u;
    std::vector<Tsetlini::real_type> accuracy_test(ensemble_size);

    std::vector<int> ix(df_X.size());
    std::iota(ix.begin(), ix.end(), 0);
    auto const seed = std::random_device{}();
    std::mt19937 gen(seed);

    std::size_t const PIVOT = df_X.size() * 0.8;
    std::vector<bit_vector_uint64> train_X(PIVOT);
    std::vector<float> train_y(PIVOT);
    std::vector<int> train_yi;
    std::vector<bit_vector_uint64> test_X(df_X.size() - PIVOT);
    std::vector<float> test_y(df_y.size() - PIVOT);
    std::vector<int> test_yi;


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

    auto constexpr T = 5000;

    for (auto it = 0u; it < ensemble_size; ++it)
    {
        std::cout << "ENSEMBLE " << it + 1 << '\n';

        std::shuffle(ix.begin(), ix.end(), gen);

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

        Scaler scaler(T);
        scaler.fit(train_y);
        train_yi = scaler.transform(train_y);
        test_yi = scaler.transform(test_y);

        auto reg = Tsetlini::make_regressor_bitwise(R"({
            "threshold": )" + std::to_string(T) + R"(,
            "s": 2.75,
            "number_of_regressor_clauses": 2000,
            "number_of_states": 127,
            "boost_true_positive_feedback": 1,
            "random_state": 1,
            "n_jobs": 2,
            "clause_output_tile_size": 16,
            "verbose": false
        })").leftMap(error_printer)
            .rightMap([&](auto && reg)
            {
                auto const time0 = now();

                auto status = reg.fit(train_X, train_yi, 30);

                auto const elapsed = as_ms(now() - time0);

                reg.predict(test_X)
                    .leftMap(error_printer)
                    .rightMap([&](auto const & yi_hat)
                    {
                        auto const y_hat = scaler.inverse_transform(yi_hat);

                        auto const e = rms(y_hat, test_y);

                        accuracy_test[it] = e;

                        auto [mean, stdev] = stdev_mean(accuracy_test.data(), accuracy_test.data() + it + 1);
                        printf("Average RMSD on test data: %.3f +/- %.3f (%.2fs)\n", mean, 1.96 * stdev / std::sqrt(it + 1), elapsed);

                        return yi_hat;
                    });

                return reg;
            });

    }

    return 0;
}
