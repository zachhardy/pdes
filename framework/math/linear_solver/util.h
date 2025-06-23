#pragma once
#include "framework/math/matrix.h"

namespace pdes
{
  namespace internal
  {
    template<typename Number>
    Vector<Number>
    extract_inv_diagonal(const Matrix<Number>& matrix, const std::string& label)
    {
      if (matrix.m() != matrix.n())
        throw std::invalid_argument(label + ": Matrix must be square.");

      Vector<Number> result(matrix.m());
      for (size_t i = 0; i < matrix.m(); ++i)
      {
        const auto diag = matrix(i, i);
        if (diag == Number(0))
          throw std::runtime_error(label + ": Zero diagonal in row " + std::to_string(i) + ".");
        result(i) = Number(1) / diag;
      }
      return result;
    }
  }
}
