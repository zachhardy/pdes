#include "framework/mesh/orthomesh_generator.h"
#include "framework/logger.h"
#include <iostream>

#include "framework/math/linear_solver/jacobi_solver.h"


int
main()
{
  std::cout << "Welcome to PDEs!" << std::endl;

  using namespace pdes;

  constexpr auto nx = 10;
  std::vector<double> x_verts;
  for (int i = 0; i < nx + 1; ++i)
    x_verts.emplace_back(static_cast<double>(i));

  constexpr auto ny = 10;
  std::vector<double> y_verts;
  for (int j = 0; j < ny + 1; ++j)
    y_verts.emplace_back(static_cast<double>(j));

  const auto mesh = OrthoMeshGenerator::create_2d_orthomesh(x_verts, y_verts);

  double total = 0.0;
  for (const auto& cell: mesh.cells())
    total += cell.volume();
  std::cout << "Total volume is " << total << std::endl;
}
