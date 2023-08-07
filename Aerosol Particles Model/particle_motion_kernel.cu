
/* Particle Movement kernel ---------------------
*     Moves the particles in each iteration
*     Each thread handles 1 particle only
*/
#include "aerosol_can.h"

__global__
void particle_motion_kernel(AerosolCan* device_particles, int num_particles, double DT) {

    // Calculate thread rank
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Move the particle
    if (index < num_particles) {
        // Update the velocity vector of the particle
        device_particles[index].vel.x += device_particles[index].acc.x * DT;
        device_particles[index].vel.y += device_particles[index].acc.y * DT;
        device_particles[index].vel.z += device_particles[index].acc.z * DT;

        // Update the position vector of the particle
        device_particles[index].pos.x += device_particles[index].vel.x * DT;
        device_particles[index].pos.y += device_particles[index].vel.y * DT;
        device_particles[index].pos.z += device_particles[index].vel.z * DT;

        // Check if the particle has hit the ground
        if (device_particles[index].pos.z <= 0.0) {
            device_particles[index].pos.z = 0.0;
            device_particles[index].vel.x = 0.0;
            device_particles[index].vel.y = 0.0;
            device_particles[index].vel.z = 0.0;
        }
    }

} // End Particle Movement Kernel //