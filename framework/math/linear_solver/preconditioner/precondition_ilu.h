#pragma once
#include "framework/math/matrix.h"
#include "framework/math/vector.h"

namespace pdes
{
  /**
   * Incomplete LU (ILU(0)) preconditioner.
   *
   * Performs an incomplete LU factorization of the matrix with zero fill-in
   * and applies it via forward and backward substitution.
   *
   * @tparam MatrixType Type of matrix to precondition.
   */
  template<typename MatrixType = Matrix<>>
  class PreconditionILU
  {
  public:
    using value_type = typename MatrixType::value_type;

    /// Constructs and factorizes the matrix A.
    explicit PreconditionILU(const MatrixType* A);

    /// Applies the preconditioner: solves LU z = r.
    template<typename VectorType>
    void vmult(const VectorType& src, VectorType& dst) const;

    /// Returns the name of the preconditioner.
    static std::string name() { return "PreconditionILU"; }

  private:
    const MatrixType* A_ = nullptr;
    MatrixType LU_;
  };

  /*-------------------- member functions --------------------*/

  template<typename MatrixType>
  PreconditionILU<MatrixType>::PreconditionILU(const MatrixType* A)
    : A_(A)
  {
    if (not A_->is_square())
      throw std::invalid_argument(name() + ": matrix must be square");

    LU_ = *A_;
    const auto n = LU_.m();
    for (size_t k = 0; k < n; ++k)
    {
      auto diag = LU_(k, k);
      if (diag == value_type(0))
        continue; // Skip division by zero

      for (size_t i = k + 1; i < n; ++i)
      {
        if (LU_(i, k) != value_type(0))
        {
          LU_(i, k) /= diag;
          for (int j = k + 1; j < n; ++j)
            LU_(i, j) -= LU_(i, k) * LU_(k, j);
        }
      }
    }
  }

  template<typename MatrixType>
  template<typename VectorType>
  void
  PreconditionILU<MatrixType>::vmult(const VectorType& src, VectorType& dst) const
  {
    const auto n = static_cast<int>(LU_.m());
    dst.reinit(n);

    // Forward substitution (Ly = src)
    for (size_t i = 0; i < n; ++i)
    {
      dst(i) = src(i);
      for (int j = 0; j < i; ++j)
        dst(i) -= LU_(i, j) * dst(j);
    }

    // Backward substitution (Ux = y)
    for (int i = n - 1; i >= 0; --i)
    {
      for (size_t j = i + 1; j < n; ++j)
        dst(i) -= LU_(i, j) * dst(j);
      dst(i) /= LU_(i, i);
    }
  }
}
