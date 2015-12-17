#ifndef NLP_DEFAULT_TYPES_HPP_
#define NLP_DEFAULT_TYPES_HPP_

#include <limits>
#include <vector>

namespace nlp {

using std::numeric_limits;
using std::vector;

using default_floating_point_type = float;
using default_integer_type = int;

template <typename F>
struct epsilon {
  static const F value;
};

template <typename F>
const F epsilon<F>::value = numeric_limits<F>::epsilon();

template <typename F=default_floating_point_type>
struct default_storage_type {
  using value = vector<F>;
};

template <typename I=default_integer_type>
struct default_index_type {
  using value = vector<I>;
};

}  // namespace nlp

#endif  // NLP_DEFAULT_TYPES_HPP_
