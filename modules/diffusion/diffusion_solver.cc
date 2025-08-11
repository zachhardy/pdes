#include "modules/diffusion/diffusion_solver.h"

namespace pdes
{
  DiffusionSolver::DiffusionSolver(
    const DiffusionModel& model,
    const std::shared_ptr<SpatialDiscretization>& discretization,
    const std::shared_ptr<LinearSolver<SparseMatrix<>>>& solver,
    const std::shared_ptr<Preconditioner<SparseMatrix<>>>& preconditioner)
    : model_(model),
      discretization_(discretization),
      solver_(solver),
      preconditioner_(preconditioner)
  {
    const auto n = discretization_->num_local_nodes();
    A_.reinit(n, n);
    b_.reinit(n);
    x_.reinit(n);

    assemble();
    preconditioner_->build(&A_);
  }

  void
  DiffusionSolver::assemble()
  {
    const auto& mesh = discretization_->mesh();
    if (discretization_->type() == SpatialDiscretizationType::FINITE_VOLUME)
    {
      for (const auto& cell: mesh->local_cells())
      {
        const auto imap = cell.local_id();

        const auto kp = model_.k();
        const auto centroid_p = cell.centroid();

        // Add source function
        b_.add(imap, model_.q(centroid_p) * cell.volume());

        for (const auto& face: cell.faces())
        {
          const auto& area = face.area();
          const auto& centroid_f = face.centroid();
          const auto& nhat = face.normal();
          const auto dpf = centroid_f - centroid_p;

          if (face.has_neighbor())
          {
            const auto& nbr_cell = mesh->local_cell(face.neighbor_id());
            const auto jmap = nbr_cell.local_id();

            const auto kn = model_.k();
            const auto& centroid_n = nbr_cell.centroid();
            const auto dnf = centroid_n - centroid_f;

            const auto dp = dpf.dot(nhat);
            const auto dn = dnf.dot(nhat);
            const auto kf = (dp + dn) / (dp / kp + dn / kn);

            const auto coeff = kf * area / (dp + dn);
            A_.add(imap, imap, coeff);
            A_.add(imap, jmap, -coeff);
          } // if interior face
          else
          {
            const auto& bid = face.boundary_id();
            const auto& bc = model_.bc(bid);
            if (bc.type == DiffusionModel::BoundaryCondition::Type::DIRICHLET)
            {
              const auto coeff = kp * area / dpf.dot(nhat);
              A_.add(imap, imap, coeff);
              b_.add(imap, coeff * bc.dirichlet.f);
            }
            else if (bc.type == DiffusionModel::BoundaryCondition::Type::NEUMANN)
            {
              b_.add(imap, area * bc.neumann.g);
            }
            else if (bc.type == DiffusionModel::BoundaryCondition::Type::ROBIN)
            {
              A_.add(imap, imap, bc.robin.a / bc.robin.b * area);
              b_.add(imap, bc.robin.f / bc.robin.b * area);
            }
          } // if boundary face
        } // for face
      } // for cell
    } // if finite volume
    A_.compress();
  }

  void
  DiffusionSolver::solve()
  {
    solver_->solve(A_, b_, x_, *preconditioner_);
  }
}
