#include "framework/math/vector.h"
#include "framework/math/matrix.h"
#include "framework/mesh/orthomesh_generator.h"
#include <iostream>


int
main()
{
  std::cout << "Welcome to PDEs!" << std::endl;

  using namespace pdes;

  Matrix<> A(3, 3, 0.0);
  for (size_t i = 0; i < A.m(); ++i)
    for (size_t j = 0; j < A.n(); ++j)
    {
      A(i, j) = static_cast<double>(i * A.n() + j + 1);
    }
  A.print();

  Vector<> x(3, 0.0);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] += static_cast<double>(i + 1);
  x.print();

  const auto b = A * x;
  b.print();

  constexpr double width = 20.0;
  constexpr unsigned int nx = 10;

  std::vector<double> x_coords;
  x_coords.reserve(nx + 1);
  for (unsigned int i = 0; i < nx + 1; ++i)
    x_coords.push_back(i * width / nx);

  auto mesh = create_1d_orthomesh(x_coords);
  for (const auto& vertex: mesh.vertices())
    vertex.print();
}
