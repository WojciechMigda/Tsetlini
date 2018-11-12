#pragma once

#include "tsetlin_config.hpp"
#include "tsetlin_types.hpp"
#include "tsetlin_state.hpp"

#include <vector>


namespace Tsetlin
{


struct Classifier
{
    Classifier(ClassifierState const & state);
    Classifier(ClassifierState && state);

    void fit_batch(std::vector<aligned_vector_char> const & X, y_vector_type const & y);

    void fit(std::vector<aligned_vector_char> const & X, y_vector_type const & y, std::size_t number_of_examples, int epochs=100);

    real_type evaluate(std::vector<aligned_vector_char> const & X, y_vector_type const & y, int number_of_examples);

    int predict(aligned_vector_char const & sample);

    aligned_vector_int predict_raw(aligned_vector_char const & sample);

    void predict_raw(aligned_vector_char const & sample, int * out_p);

private:
    ClassifierState state;
};


} // namespace Tsetlin
