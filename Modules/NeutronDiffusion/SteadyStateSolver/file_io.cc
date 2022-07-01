#include "steadystate_solver.h"

#include <fstream>
#include <cstring>
#include <cassert>


using namespace NeutronDiffusion;


void
SteadyStateSolver::write_snapshot(const std::string& file_base) const
{
  std::string file_name = file_base + ".data";

  // Open the file
  std::ofstream file(file_name,
                     std::ofstream::binary |
                     std::ofstream::out |
                     std::ofstream::trunc);
  assert(file.is_open());

  // Write the header_info
  int size = 800;
  std::string header_info =
    "NeutronDiffusion::SteadyStateSolver: Snapshot File\n"
    "Header size: " + std::to_string(size) + " bytes\n";
  header_info +=
    "Structure(type-info):\n"
    "unsigned int  n_data_blocks\n"
    "uint64_t      n_cells\n"
    "uint64_t      n_nodes\n"
    "unsigned int  n_moments\n"
    "unsigned int  n_groups\n"
    "unsigned int  n_precursors\n"
    "unsigned int  max_precursors\n"
    "Each Cell:\n"
    "  uint64_t      cell_id\n"
    "  uint64_t      material_id\n"
    "  unsigned int  n_nodes\n"
    "  Centroid:\n"
    "    double  centroid_x_position\n"
    "    double  centroid_y_position\n"
    "    double  centroid_z_position\n"
    "  Each node:\n"
    "    double  x_position\n"
    "    double  y_position\n"
    "    double  z_position\n"
    "Scalar Flux Records:\n"
    "  unsigned int  record_type\n"
    "  uint64_t      n_records\n"
    "  Each Record:\n"
    "    uint64_t      cell_id\n"
    "    unsigned int  node\n"
    "    unsigned int  moment\n"
    "    unsigned int  group\n"
    "    double        value\n"
    "Precursor Records:\n"
    "  unsigned int  record_type\n"
    "  uint64_t      n_records\n"
    "  Each Record:\n"
    "    uint64_t      cell_id\n"
    "    uint64_t      material_id\n"
    "    unsigned int  precursor\n"
    "    double        value\n";

  int header_size = (int)header_info.length();

  char header_bytes[size];
  memset(header_bytes, '-', size);
  strncpy(header_bytes, header_info.c_str(), std::min(header_size, size - 1));
  header_bytes[size - 1] = '\0';

  const unsigned int n_data_blocks = (use_precursors)? 2 : 1;
  uint64_t n_records;
  unsigned int record_type = 0;

  const uint64_t n_cells = mesh->cells.size();
  const uint64_t n_nodes = discretization->nodes_per_cell();

  const unsigned int n_moments = 1;

  // Write header_info and general information
  file << header_bytes;
  file.write((char*)&n_records, sizeof(unsigned int));

  file.write((char*)&n_cells, sizeof(uint64_t));
  file.write((char*)&n_nodes, sizeof(uint64_t));

  file.write((char*)&n_moments, sizeof(unsigned int));
  file.write((char*)&n_groups, sizeof(unsigned int));
  file.write((char*)&n_precursors, sizeof(unsigned int));
  file.write((char*)&max_precursors, sizeof(unsigned int));

  // Write discretization information
  for (const auto& cell : mesh->cells)
  {
    const uint64_t cell_id = cell.id;
    const uint64_t material_id = cell.material_id;

    const auto& centroid = cell.centroid;

    const auto& nodes = discretization->nodes(cell);
    const unsigned int npc = nodes.size();

    file.write((char*)&cell_id, sizeof(uint64_t));
    file.write((char*)&material_id, sizeof(uint64_t));
    file.write((char*)&npc, sizeof(unsigned int));

    // Write the centroid
    file.write((char*)&centroid.x, sizeof(double));
    file.write((char*)&centroid.y, sizeof(double));
    file.write((char*)&centroid.z, sizeof(double));

    // Write the nodes
    for (const auto& node : nodes)
    {
      file.write((char*)&node.x, sizeof(double));
      file.write((char*)&node.y, sizeof(double));
      file.write((char*)&node.z, sizeof(double));
    }
  }//for cell

  // Write scalar flux data
  n_records = phi.size();

  file.write((char*)&record_type, sizeof(unsigned int));
  file.write((char*)&n_records, sizeof(uint64_t));

  for (const auto& cell : mesh->cells)
  {
    const uint64_t cell_id = cell.id;
    const unsigned int npc = discretization->nodes(cell).size();

    for (unsigned int i = 0; i < npc; ++i)
      for (unsigned int m = 0; m < n_moments; ++m)
        for (unsigned int g = 0; g < n_groups; ++g)
        {
          const uint64_t dof = cell.id*npc*n_groups + i*n_groups + g;
          assert(dof < phi.size());

          file.write((char*)&cell_id, sizeof(uint64_t));
          file.write((char*)&i, sizeof(unsigned int));
          file.write((char*)&m, sizeof(unsigned int));
          file.write((char*)&g, sizeof(unsigned int));
          file.write((char*)&phi[dof], sizeof(double));
        }
  }//for cell
  ++record_type;

  // Write precursor data
  if (use_precursors)
  {
    n_records = precursors.size();

    file.write((char*)&record_type, sizeof(unsigned int));
    file.write((char*)&n_records, sizeof(uint64_t));

    for (const auto& cell : mesh->cells)
    {
      const uint64_t cell_id = cell.id;
      const uint64_t material_id = cell.material_id;

      const auto& xs = material_xs[matid_to_xs_map[material_id]];
      for (unsigned int j = 0; j < xs->n_precursors; ++j)
      {
        const uint64_t dof = cell.id*max_precursors + j;
        assert(dof < precursors.size());

        file.write((char*)&cell_id, sizeof(uint64_t));
        file.write((char*)&material_id, sizeof(uint64_t));
        file.write((char*)&j, sizeof(unsigned int));
        file.write((char*)&precursors[dof], sizeof(double));
      }
    }//for cell
  }

  file.close();
}