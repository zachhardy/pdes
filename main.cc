#include "modules/diffusion/diffusion_model.h"
#include "modules/diffusion/diffusion_solver.h"
#include "framework/mesh/orthomesh_generator.h"
#include "framework/math/spatial_discretization/finite_volume.h"
#include <iostream>


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
  const auto fv = std::make_shared<FiniteVolume>(mesh);


  const auto model = DiffusionModel(1.0,
                                    [](const MeshVector<>&) { return 1.0; });
  auto solver = DiffusionSolver(model, fv);
  solver.assemble();
}
