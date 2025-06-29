#pragma once
#include "framework/math/matrix.h"

namespace pdes
{
  namespace internal
  {
    /**
     * Computes the inverse of diagonal entries from a square matrix.
     *
     * This function is used in iterative solvers to construct simple
     * preconditioners based on diagonal scaling. Throws if the matrix is not
     * square or contains a zero on the diagonal.
     *
     * @tparam Number Scalar type used in the matrix.
     * @param matrix A square matrix from which to extract the diagonal.
     * @param label Descriptive label used in exception messages.
     * @return Vector containing 1 / diagonal(i) for each row.
     */
    template<typename Number>
    std::vector<Number>
    extract_inv_diagonal(const Matrix<Number>& matrix, const std::string& label)
    {
      if (matrix.m() != matrix.n())
        throw std::invalid_argument(label + ": Matrix must be square.");

      std::vector<Number> result;
      for (size_t i = 0; i < matrix.m(); ++i)
      {
        const auto diag = matrix(i, i);
        if (diag == Number(0))
          throw std::runtime_error(label + ": Zero diagonal in row " + std::to_string(i) + ".");
        result.push_back(Number(1) / diag);
      }
      return result;
    }
  }
}
