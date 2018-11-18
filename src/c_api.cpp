#include "tsetlin.hpp"

#include "tsetlin_types.hpp"
#include "config_companion.hpp"

#undef LOG_MODULE
#define LOG_MODULE "tsetlin"
#include "logger.hpp"

#include <new>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <string>
#include <cstring>
#include <random>


extern "C"
{


namespace
{

using tsetlin_classifier_p = Tsetlin::Classifier const *;
using tsetlin_classifier_mut_p = Tsetlin::Classifier *;

using aligned_vector_char = Tsetlin::aligned_vector_char;
using aligned_vector_int = Tsetlin::aligned_vector_int;

tsetlin_classifier_p TsetlinClassifierPtr(void * handle)
{
    return static_cast<tsetlin_classifier_p>(handle);
}

tsetlin_classifier_mut_p TsetlinClassifierMutPtr(void * handle)
{
    return static_cast<tsetlin_classifier_mut_p>(handle);
}

}


void * TsetlinClassifier(std::string const & json_config)
{
    auto state = Tsetlin::make_classifier_state(Tsetlin::config_patch_from_json(json_config));

    auto ptr = new (std::nothrow) Tsetlin::Classifier(state);

    return ptr;
}


void TsetlinClassifierFree(void * handle)
{
    delete TsetlinClassifierPtr(handle);
}


int TsetlinMachineFitCSR(
    void * handle,
    const std::size_t * indptr,
    const unsigned int * indices,
    const char * X_p,
    std::size_t nindptr,
    std::size_t nelem,
    std::size_t num_col,
    const int * y_p,
    std::size_t nepochs,
    std::size_t batch_sz,
    char verbose
    )
{
    auto self = TsetlinClassifierMutPtr(handle);

#if 0
    std::cout << "TsetlinClassifierFitCSR\n";
    std::cout << "nindptr " << nindptr << '\n';
    std::cout << "nelem " << nelem << '\n';
    std::cout << "num_col " << num_col << '\n';
    std::cout << "nepochs " << nepochs << '\n';
    std::cout << "batch_sz " << batch_sz << '\n';
    std::cout << "y " << y_p << '\n';
    std::copy_n(y_p, 10, std::ostream_iterator<int>(std::cout, " ")); std::cout << '\n';
#endif

    auto const seed = Tsetlin::Config::seed(self->read_config());
    std::mt19937 gen(seed);

    const auto n_rows = nindptr - 1;
//    const auto n_batches = (n_rows + batch_sz - 1) / batch_sz;

    std::vector<int> ix(n_rows);
    std::iota(ix.begin(), ix.end(), 0);

    std::vector<int> batches;
    for (auto ix = 0u; ix < n_rows; ix += batch_sz)
    {
        batches.push_back(ix);
    }
    batches.push_back(n_rows);

    std::vector<int> y;
    std::vector<aligned_vector_char> X(batch_sz, aligned_vector_char(num_col));

    for (auto epoch = 0u; epoch < nepochs; ++epoch)
    {
        LOG(info) << "FitCSR: epoch " << epoch + 1 << " of " << nepochs << '\n';

        std::shuffle(ix.begin(), ix.end(), gen);

        for (auto batches_it = 0u; batches_it < batches.size() - 1; ++batches_it)
        {
            LOG(info) << "FitCSR: batch " << batches_it + 1 << " of " << batches.size() - 1 << '\n';

            auto const batch_begin = batches[batches_it];
            auto const batch_end = batches[batches_it + 1];

            // decode CSR into vector input for Tsetlin Classifier

            X.resize(batch_end - batch_begin, aligned_vector_char(num_col));
            y.resize(batch_end - batch_begin);

            for (auto batch_it = batch_begin; batch_it < batch_end; ++batch_it)
            {
                auto const irow = ix[batch_it];
                auto const orow = batch_it - batch_begin;

                y[orow] = y_p[irow];

                std::memset(X[orow].data(), 0, num_col);
                for (auto ind_it = indptr[irow]; ind_it < indptr[irow + 1]; ++ind_it)
                {
                    auto const column = indices[ind_it];
                    auto const value = X_p[ind_it];
                    X[orow][column] = value;
                }
#if 0
                std::cout << irow << '\n';
                std::copy(X[orow].cbegin(), X[orow].cend(), std::ostream_iterator<int>(std::cout, " "));
                std::cout << (int)y[orow] << '\n';
#endif
            }

            self->fit_batch(X, y);
        }
    }

    return 0;
}


int TsetlinClassifierPredictCSR(
    void * handle,
    const std::size_t * indptr,
    const unsigned int * indices,
    const char * X_p,
    std::size_t nindptr,
    std::size_t nelem,
    std::size_t num_col,
    std::uint64_t * length_p,
    char * yhat_p,
    char verbose
    )
{
    auto self = TsetlinClassifierMutPtr(handle);

#if 0
    std::cout << "TsetlinClassifierPredictCSR\n";
    std::cout << "nindptr " << nindptr << '\n';
    std::cout << "nelem " << nelem << '\n';
    std::cout << "num_col " << num_col << '\n';
    std::cout << "*length_p " << *length_p << '\n';
    std::cout << "yhat_p " << yhat_p << '\n';
    std::copy_n(yhat_p, std::min<std::uint64_t>(10, *length_p), std::ostream_iterator<int>(std::cout, " ")); std::cout << '\n';
#endif

    const auto n_rows = nindptr - 1;

    aligned_vector_char X(num_col);

    for (auto rit = 0u; rit < n_rows; ++rit)
    {
        std::memset(X.data(), 0, X.size() * sizeof (X[0]));

        for (auto ind_it = indptr[rit]; ind_it < indptr[rit + 1]; ++ind_it)
        {
            auto const column = indices[ind_it];
            auto const value = X_p[ind_it];
            X[column] = value;
        }

        yhat_p[rit] = self->predict(X);

#if 0
        std::cout << rit << '\n';
        std::copy(X.cbegin(), X.cend(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << (int)yhat_p[rit] << '\n';
#endif
    }

    *length_p = n_rows;

    return 0;
}


int TsetlinMachinePredictProba1CSR(
    void * handle,
    const std::size_t * indptr,
    const unsigned int * indices,
    const char * X_p,
    std::size_t nindptr,
    std::size_t nelem,
    std::size_t num_col,
    std::uint64_t * length_p,
    char * yhat_p,
    float * proba_p,
    char verbose
    )
{
    auto self = TsetlinClassifierMutPtr(handle);

    const auto n_rows = nindptr - 1;

    aligned_vector_char X(num_col);

    auto const nclasses = Tsetlin::Config::number_of_classes(self->read_config());
    aligned_vector_int counts(nclasses);
    std::vector<float> probas(nclasses);

    for (auto rit = 0u; rit < n_rows; ++rit)
    {
        std::memset(X.data(), 0, X.size() * sizeof (X[0]));

        for (auto ind_it = indptr[rit]; ind_it < indptr[rit + 1]; ++ind_it)
        {
            auto const column = indices[ind_it];
            auto const value = X_p[ind_it];
            X[column] = value;
        }

        self->predict_raw(X, counts.data());

        yhat_p[rit] = std::distance(counts.cbegin(), std::max_element(counts.cbegin(), counts.cend()));

        std::transform(counts.cbegin(), counts.cend(), probas.begin(), [](int cnt){ return std::exp((float(cnt))); });
        auto const sigma = std::accumulate(probas.cbegin(), probas.cend(), 0.0f);

        proba_p[rit] = probas[yhat_p[rit]] / sigma;
    }

    *length_p = n_rows;

    return 0;
}



} // extern "C"
