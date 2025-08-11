#include "modules/diffusion/diffusion_model.h"
#include "modules/diffusion/diffusion_solver.h"
#include "framework/mesh/orthomesh_generator.h"
#include "framework/math/linear_solver/cg_solver.h"
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

  const auto mesh = OrthoMeshGenerator::create_2d_orthomesh(x_verts, x_verts);
  const auto fv = std::make_shared<FiniteVolume>(mesh);

  auto k = [](const MeshVector<>&) { return 1.0; };
  std::map<unsigned int, DiffusionModel::BoundaryCondition> bcs;
  for (unsigned int b = 0; b < 6; ++b)
  {
    auto bc = DiffusionModel::BoundaryCondition{};
    bc.type = DiffusionModel::BoundaryCondition::Type::DIRICHLET;
    bc.dirichlet = DiffusionModel::BoundaryCondition::Dirichlet{0.0};
    bcs[b] = bc;
  }

  auto logger = Logger();
  logger.set_level(LogLevel::ITERATION);

  auto control = SolverControl(1000, 1.0e-8);
  auto linsol = std::make_shared<CGSolver<SparseMatrix<>>>(&control);
  linsol->set_logger(logger);
  auto M = std::make_shared<PreconditionSSOR<SparseMatrix<>>>();
  const auto model = DiffusionModel(1.0, k, bcs);
  auto solver = DiffusionSolver(model, fv, linsol, M);
  solver.solve();

  const auto& phi = solver.solution();
  std::cout << "Solution has " << phi.size() << " entries." << std::endl;
}
