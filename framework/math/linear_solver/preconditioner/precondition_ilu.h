#pragma once
#include "framework/math/matrix.h"
#include "framework/math/vector.h"

namespace pdes
{
  /**
   * Computes an incomplete LU factorization with no fill-in.
   * Solves z = Ainv r via forward and backward substitution.
   */
  template<typename MatrixType>
  class PreconditionILU
  {
  public:
    using value_type = typename MatrixType::value_type;

    explicit PreconditionILU(const MatrixType* A);

    template<typename VectorType>
    void vmult(const VectorType& src, VectorType& dst) const;

    static std::string name() { return "PreconditionILU"; }

  private:
    const MatrixType* A_ = nullptr;
    MatrixType L_, U_;
  };

  /*-------------------- inline functions --------------------*/

  template<typename MatrixType>
  PreconditionILU<MatrixType>::PreconditionILU(const MatrixType* A)
    : A_(A)
  {
    if (not A_->is_square())
      throw std::invalid_argument(name() + ": matrix must be square");

    const auto n = A_->m();
    L_ = Matrix<value_type>(n, n, value_type(0));
    U_ = Matrix<value_type>(n, n, value_type(0));

    for (size_t i = 0; i < n; ++i)
    {
      for (size_t j = 0; j < n; ++j)
      {
        value_type sum = (*A_)(i, j);
        for (size_t k = 0; k < std::min(i, j); ++k)
          sum -= L_(i, k) * U_(k, j);

        if (i > j)
          L_(i, j) = sum / U_(j, j);
        else
        {
          U_(i, j) = sum;
          if (i == j)
            L_(i, i) = value_type(1);
        }
      }
    }
  }

  template<typename MatrixType>
  template<typename VectorType>
  void
  PreconditionILU<MatrixType>::vmult(const VectorType& src, VectorType& dst) const
  {
    if (not A_)
      throw std::runtime_error(name() + " not initialized");
    const auto n = src.size();
    if (n != L_.m())
      throw std::runtime_error(name() + ": size mismatch");

    Vector<value_type> y(n, value_type(0));
    dst.resize(n);

    // Forward solve Ly = src
    for (size_t i = 0; i < n; ++i)
    {
      value_type sum = src(i);
      for (size_t j = 0; j < i; ++j)
        sum -= L_(i, j) * y(j);
      y(i) = sum;
    }

    // Backward solve Ux = y
    for (size_t i = n; i-- > 0;)
    {
      value_type sum = y(i);
      for (size_t j = i + 1; j < n; ++j)
        sum -= U_(i, j) * dst(j);

      if (U_(i, i) == value_type(0))
        throw std::runtime_error(name() + ": zero pivot at row " + std::to_string(i));

      dst(i) = sum / U_(i, i);
    }
  }
}
