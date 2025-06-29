#pragma once
#include "framework/math/linear_solver/preconditioner/preconditioner.h"

namespace pdes
{
  template<typename Number = types::real>
  class PreconditionIdentity final : public Preconditioner<Number>
  {
  public:
    void vmult(const Vector<Number>& src, Vector<Number>& dst) const override;

    std::string name() const override { return "PreconditionIdentity"; }
  };

  /*-------------------- inline functions --------------------*/

  template<typename Number>
  void
  PreconditionIdentity<Number>::vmult(const Vector<Number>& src, Vector<Number>& dst) const
  {
    dst = src;
  }
}
