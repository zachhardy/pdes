#pragma once
#include "framework/math/matrix.h"
#include <string>

namespace pdes
{
  template<typename MatrixType = Matrix<>>
  class LUSolver
  {
  public:
    using value_type = typename MatrixType::value_type;

    LUSolver() = default;

    void factor(const MatrixType& A);

    template<typename VectorType>
    void solve(const MatrixType& A, const VectorType& b, VectorType& x) const;

    template<typename VectorType>
    void solve(const VectorType& b, VectorType& x) const;

    static std::string name() { return "LUSolver"; }

  private:
    MatrixType L_, U_;
    bool factored_ = false;
  };

  /*-------------------- inline functions --------------------*/

  template<typename MatrixType>
  void
  LUSolver<MatrixType>::factor(const MatrixType& A)
  {
    const size_t n = A.m();
    if (A.m() != A.n())
      throw std::invalid_argument(name() + ": matrix must be square.");

    L_ = MatrixType(n, n, value_type(0));
    U_ = A;

    for (size_t k = 0; k < n; ++k)
    {
      if (U_(k, k) == value_type(0))
        throw std::runtime_error(name() + ": zero pivot at row " + std::to_string(k));

      L_(k, k) = value_type(1);
      for (size_t i = k + 1; i < n; ++i)
      {
        const auto m = U_(i, k) / U_(k, k);
        L_(i, k) = m;
        for (size_t j = k; j < n; ++j)
          U_(i, j) -= m * U_(k, j);
        U_(i, k) = 0; // optional: keep upper matrix clean
      }
    }
    factored_ = true;
  }

  template<typename MatrixType>
  template<typename VectorType>
  void
  LUSolver<MatrixType>::solve(const MatrixType& A, const VectorType& b, VectorType& x) const
  {
    factor(A);
    solve(b, x);
  }

  template<typename MatrixType>
  template<typename VectorType>
  void
  LUSolver<MatrixType>::solve(const VectorType& b, VectorType& x) const
  {
    if (not factored_)
      throw std::runtime_error(name() + ": matrix not factorized");
    if (b.size() != L_.m())
      throw std::invalid_argument(name() + ": size of b does not match factorized system");

    const size_t n = L_.m();

    // Forward solve: L y = b
    VectorType y(n);
    for (size_t i = 0; i < n; ++i)
    {
      y(i) = b(i);
      for (size_t j = 0; j < i; ++j)
        y(i) -= L_(i, j) * y(j);
    }

    // Backward solve: U x = y
    x.resize(n);
    for (size_t i = n; i-- > 0;)
    {
      x(i) = y(i);
      for (size_t j = i + 1; j < n; ++j)
        x(i) -= U_(i, j) * x(j);
      x(i) /= U_(i, i);
    }
  }
}
