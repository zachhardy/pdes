#pragma once
#include "framework/math/linear_solver/preconditioner.h"
#include "framework/math/matrix.h"


namespace pdes
{
  template<typename Number = double>
  class PreconditionJacobi : public Preconditioner<Number>
  {
  public:
    PreconditionJacobi() = default;

    void initialize(const Matrix<Number>& A);

    void vmult(const Vector<Number>& src, Vector<Number>& dst) const override;

    std::string name() const override { return "PreconditionJacobi"; }

  private:
    const Matrix<Number>* A_;
    std::vector<Number> inv_diag_;
  };

  /*-------------------- inline functions --------------------*/

  template<typename Number>
  void
  PreconditionJacobi<Number>::initialize(const Matrix<Number>& A)
  {
    A_ = &A;

    const size_t n = A.m();
    inv_diag_.assign(n, Number(0));

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
  PreconditionJacobi<Number>::vmult(const Vector<Number>& src, Vector<Number>& dst) const
  {
    if (not A_)
      throw std::runtime_error(name() + " not initialized");
    if (inv_diag_.size() != src.size())
      throw std::runtime_error(name() + ": dimension mismatch error");

    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
      dst(i) = inv_diag_[i] * src(i);
  }
}
