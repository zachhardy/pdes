#pragma once
#include "framework/math/linear_solver/preconditioner.h"
#include "framework/math/matrix.h"

namespace pdes
{
  /**
   * SOR preconditioner:
   * Applies z ≈ A⁻¹ r via a forward + backward sweep.
   */
  template<typename Number = double>
  class PreconditionSSOR final : public Preconditioner<Number>
  {
  public:
    explicit PreconditionSSOR(Number omega = 1.3);

    void initialize(const Matrix<Number>& A);

    void vmult(const Vector<Number>& src, Vector<Number>& dst) const override;

    std::string name() const override { return "PreconditionSSOR"; }

  private:
    Number omega_;
    const Matrix<Number>* A_ = nullptr;
    std::vector<Number> inv_diag_;
  };

  /*-------------------- inline functions --------------------*/

  template<typename Number>
  PreconditionSSOR<Number>::PreconditionSSOR(Number omega)
    : omega_(omega)
  {
    if (omega <= 0.0 or omega >= 2.0)
      throw std::invalid_argument("SOR omega must be in (0, 2)");
  }

  template<typename Number>
  void
  PreconditionSSOR<Number>::initialize(const Matrix<Number>& A)
  {
    A_ = &A;

    const size_t n = A.m();
    inv_diag_.resize(n);

    for (size_t i = 0; i < n; ++i)
    {
      const auto val = A(i, i);
      if (val == Number(0))
        throw std::runtime_error(name() + ": zero diagonal at row " + std::to_string(i));
      inv_diag_[i] = Number(1) / val;
    }
  }

  template<typename Number>
  void
  PreconditionSSOR<Number>::vmult(const Vector<Number>& src, Vector<Number>& dst) const
  {
    if (not A_)
      throw std::runtime_error(name() + " not initialized");
    if (src.size() != inv_diag_.size())
      throw std::runtime_error(name() + ": size mismatch");

    const auto n = src.size();
    Vector<Number> y(n, Number(0));
    dst.resize(n, Number(0));

    // Forward solve: (D + \omega L) y = r
    for (size_t i = 0; i < n; ++i)
    {
      const Number* row = A_->begin(i);
      Number sum = 0;
      for (size_t j = 0; j < i; ++j)
        sum += row[j] * y(j); // lower triangle only

      y(i) = (src(i) - omega_ * sum) * inv_diag_[i];
    }

    // Backward solve: (D + \omega L)^T z = D y
    for (size_t i = n; i-- > 0;)
    {
      Number sum = 0;
      const auto row = A_->begin(i);
      for (size_t j = i + 1; j < n; ++j)
        sum += row[j] * dst(j); // upper triangle of transpose(L)

      const auto rhs = inv_diag_[i] * y(i);
      dst(i) = (rhs - omega_ * sum) * inv_diag_[i];
    }
  }
}
