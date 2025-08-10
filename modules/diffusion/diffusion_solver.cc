#include "modules/diffusion/diffusion_solver.h"

namespace pdes
{
  DiffusionSolver::DiffusionSolver(const DiffusionModel& model,
                                   const std::shared_ptr<SpatialDiscretization>& discretization)
    : model_(model),
      discretization_(discretization)
  {
    const auto n = discretization_->num_local_nodes();
    A_.reinit(n, n);
    b_.reinit(n);
  }

  void
  DiffusionSolver::assemble()
  {
    const auto& mesh = discretization_->mesh();
    if (discretization_->type() == SpatialDiscretizationType::FINITE_VOLUME)
    {
      for (const auto& cell: mesh->local_cells())
      {
        const auto centroid_p = cell.centroid();
        const auto imap = cell.local_id();
        const auto kp = model_.k();

        b_.add(imap, model_.q(centroid_p) * cell.volume());

        for (const auto& face: cell.faces())
        {
          const auto area = face.area();
          const auto nhat = face.normal();

          if (face.has_neighbor())
          {
            const auto& nbr_cell = mesh->local_cell(face.neighbor_id());
            const auto jmap = nbr_cell.local_id();
            const auto kn = model_.k();

            const auto dpf = face.centroid() - centroid_p;
            const auto dpn = nbr_cell.centroid() - centroid_p;
            const auto w = dpf.length() / dpn.length();
            const auto kf = 1.0 / (w / kp + (1.0 - w) / kn);
            const auto val = kf * nhat.dot(dpn) / dpn.length_sqr() * area;

            A_.add(imap, imap, val);
            A_.add(imap, jmap, -val);
          }
        }
      }
    }
    A_.compress();
  }
}
