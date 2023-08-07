
/* Paper Colouring kernel ---------------------
*     Colours the paper after particle impact
*     Each thread handles 1 particle only
*/
#include "aerosol_can.h"

__global__
void colouring_kernel(AerosolCan* device_particles, Colour* device_grid, int num_particles, int GRID_SIZE, double BLEND) {

    // Calculate thread rank
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Move the particle
    if (index < num_particles) {
        // Convert the final position of particles into grid coordinates
        int x_coord = (int)round(device_particles[index].pos.x /*+ ((((double)rand() / RAND_MAX) * 100) - 50)*/);
        int y_coord = (int)round(device_particles[index].pos.y /*+ ((((double)rand() / RAND_MAX) * 100) - 50)*/);

        //printf("%d  %d\n", x_coord, y_coord);

        // Check which particles land on the paper
        device_particles[index].check_hit = 1;
        if (x_coord < 0 || y_coord < 0 || x_coord > GRID_SIZE - 1 || y_coord > GRID_SIZE - 1)
            device_particles[index].check_hit = 0;

        // Update the colour of only those coordinates that land on the paper
        if (device_particles[index].check_hit == 1)
        {
            device_grid[x_coord * GRID_SIZE + y_coord].R = /*(1 - BLEND) + */((1 - BLEND) * device_particles[index].col.R);
            device_grid[x_coord * GRID_SIZE + y_coord].G = /*(1 - BLEND) + */((1 - BLEND) * device_particles[index].col.G);
            device_grid[x_coord * GRID_SIZE + y_coord].B = /*(1 - BLEND) + */((1 - BLEND) * device_particles[index].col.B);
        }
    }
} // End Colouring Kernel //