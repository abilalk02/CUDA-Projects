#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <driver_types.h>
#include <curand.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "particle_motion_kernel.cu"
#include "colouring_kernel.cu"
#include "aerosol_can.h"

#define PI 3.14159265358979323846           // Value of PI
#define BILLION  1000000000.0               // Used for Clock_Gettime()
#define GRID_SIZE 1000                      // Size of the grid 1000 x 1000
#define MAX_ITERATIONS 100                  // Maximum Iterations allowed
#define CAN_HEIGHT 0.2                      // Height of the can in metre
#define H_ANGLE 15.0                        // horizontal cone angle in degrees
#define V_ANGLE 15.0                        // vertical cone angle in degrees
#define VELOCITY 10.0                       // initial velocity of the particles in m/s
#define DRAG 0.5                            // Drag coefficient
#define BLEND 0.1                           // Blend coefficient
#define DT 0.1                              // time step in seconds
#define GRAVITY 9.8                         // acceleration due to gravity in m/s^2
#define CENTER_X 500                        // X coordinate of paper's center
#define CENTER_Y 500                        // Y coordinate o papers's center
#define BLOCK_SIZE 1024                     // To use with kernels

/* Function declarations */
void err_check(cudaError_t ret, char* msg, int exit_code);

/* ==========================MAIN=========================== */
int main(void)
{
    // Seed the random number generator
    //srand(time(NULL));

    // Timing and error variables
    struct timespec start, end;
    cudaError_t cuda_ret;

    // Convert the horizontal and vertical angles to radians
    double h_angle = H_ANGLE * PI / 180.0;
    double v_angle = V_ANGLE * PI / 180.0;

    // Assuming a square sheet of paper of length 10 cm (0.1 m)
    double paper_size = 0.1;

    /* The paper is represented by a grid of a 1000 x 1000 elements
       The bottom left corner of paper is placed at grid[0][0] */
    Colour* grid = (Colour*)malloc(GRID_SIZE * GRID_SIZE * sizeof(Colour));

    // The paper is initially white (1, 1, 1)
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            grid[i * GRID_SIZE + j].R = 1;
            grid[i * GRID_SIZE + j].G = 1;
            grid[i * GRID_SIZE + j].B = 1;
        }
    }
    /* Radius of the circle from the center of the sheet along which
       aerosol cans are placed. It is assumed that the square sheet
       of paper perfectly fits inside the circle. In such case, the
       required radius of circle is given by the formula below */
    
    double radius = sqrt(pow(paper_size, 2) + pow(paper_size, 2)) / 2;

    /* We don't need this radius with 4 cans because we have set the 
       grid such that cans are placed at the four corners of the sheet
       that touch the circle */

    // Simulation Parameters -----------------------------
    int num_cans = 4;                   // Number of cans
    int particles_per_can = 200000;     // Particles per can

    // Total number of particles
    int num_particles = particles_per_can * num_cans;

    // Initialize a combined array of particles emitted from all the cans
    AerosolCan* particles = (AerosolCan*)malloc(num_particles * sizeof(AerosolCan));

    // Initial Setup -----------------------------------------------------------
    // -------------------------------------------------------------------------
    // Generate the Spray particles
    for (int i = 0; i < num_cans; i++)
    {
        // Generate random numbers between 0-1 for representing colour of spray can
        double red = (double)rand() / RAND_MAX;
        double green = (double)rand() / RAND_MAX;
        double blue = (double)rand() / RAND_MAX;

        // Variables for storing X and Y coordinates of Each of the three cans
        double CAN_X, CAN_Y;
        if (i == 0)             // Can 1 is placed at coordinates (0, 0)
        {
            CAN_X = 0;
            CAN_Y = 0;
        }
        else if (i == 1)        // Can 2 is placed at coordinates (0, 1000)
        {
            CAN_X = 0;
            CAN_Y = 1000;
        }
        else if (i == 2)        // Can 3 is placed at coordinates (1000, 1000)
        {
            CAN_X = 1000;
            CAN_Y = 1000;
        }
        else if (i == 3)        // Can 4 is placed at coordinates (1000, 0)
        {
            CAN_X = 1000;
            CAN_Y = 0;
        }
        // For all particles of this can
        for (int j = 0; j < particles_per_can; j++)
        {
            // Generate horizontal and verticle angles of each of the spray particles
            // The particles exit the cans as a cone
            // The horizontal and vertical angles of cone are fixed at 15 degrees
            // rh and rv are used to convert polar angles into cartesian
            double theta = ((double)rand() / RAND_MAX) * 2.0 * PI;
            double phi = ((double)rand() / RAND_MAX) * 2.0 * PI;
            double rh = ((double)rand() / RAND_MAX) * tan(h_angle);
            double rv = ((double)rand() / RAND_MAX) * tan(v_angle);

            // Each particle's velocity will be slightly different
            // If all particles have the same velocity and same acceleration, they will all
            // reach the same distance
            double velocity = VELOCITY * ((double)rand() / RAND_MAX);

            // Calculate x, y, and z coordinates for each particle   
            particles[i * particles_per_can + j].pos.x = CAN_X + rh * cos(theta);
            particles[i * particles_per_can + j].pos.y = CAN_Y + rh * sin(theta);
            particles[i * particles_per_can + j].pos.z = (CAN_HEIGHT * 1000) + rv * sin(phi);

            // Calculate the velocity vector of the particles
            double dir_x = (CENTER_X - particles[i * particles_per_can + j].pos.x) * cos(theta);
            double dir_y = (CENTER_Y - particles[i * particles_per_can + j].pos.y) * sin(theta);
            double dir_z = 0.0 - (CAN_HEIGHT * 1000);
            double dir_mag = sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);

            // Calculate acceleration components in the x, y and z direction for this can
            double accel_x = (GRAVITY - (DRAG * pow(velocity, 2))) * dir_x / dir_mag;
            double accel_y = (GRAVITY - (DRAG * pow(velocity, 2))) * dir_y / dir_mag;
            double accel_z = (GRAVITY - (DRAG * pow(velocity, 2))) * dir_z / dir_mag;

            // Assign colour to each of the particles
            particles[i * particles_per_can + j].col.R = red;
            particles[i * particles_per_can + j].col.G = green;
            particles[i * particles_per_can + j].col.B = blue;
            particles[i * particles_per_can + j].check_hit = 0;

            // Assign velocity components in each direction
            particles[i * particles_per_can + j].vel.x = velocity * dir_x / dir_mag;
            particles[i * particles_per_can + j].vel.y = velocity * dir_y / dir_mag;
            particles[i * particles_per_can + j].vel.z = velocity * dir_z / dir_mag;

            // Assign acceleration components in each direction
            particles[i * particles_per_can + j].acc.x = accel_x;
            particles[i * particles_per_can + j].acc.y = accel_y;
            particles[i * particles_per_can + j].acc.z = accel_z;
        }
    }
    // To use with kernels
    int num_blocks = ceil((float)num_particles / (float)BLOCK_SIZE);
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    // Allocate memory for particles on device
    AerosolCan* device_particles;
    cuda_ret = cudaMalloc((void**)&device_particles, num_particles * sizeof(AerosolCan));
    err_check(cuda_ret, (char*)"Unable to allocate particles to device memory!", 1);

    // Copy particles data to device memory
    cuda_ret = cudaMemcpy(device_particles, particles, num_particles * sizeof(AerosolCan), cudaMemcpyHostToDevice);
    err_check(cuda_ret, (char*)"Unable to read particles data from host memory!", 2);

    // Allocate memory for paper grid on device
    Colour* device_grid;
    cuda_ret = cudaMalloc((void**)&device_grid, GRID_SIZE * GRID_SIZE * sizeof(Colour));
    err_check(cuda_ret, (char*)"Unable to allocate paper grid to device memory!", 3);

    // Copy paper grid to device memory
    cuda_ret = cudaMemcpy(device_grid, grid, GRID_SIZE * GRID_SIZE * sizeof(Colour), cudaMemcpyHostToDevice);
    err_check(cuda_ret, (char*)"Unable to read grid data from host memory!", 4);

    // Start clock
    clock_gettime(CLOCK_REALTIME, &start);

    // Particle Movement ------------------------------------------------------
    // ------------------------------------------------------------------------
    // Iteration Count
    int iterations = 0;
    while (iterations < MAX_ITERATIONS)
    {
        // Launch the particles movement kernel
        particle_motion_kernel <<< dimGrid, dimBlock >>> (
            device_particles,   // Particles allocated on device memory
            num_particles,      // Number of particles
            DT );              // Time step
          
        // Synchronize threads and check for error 
        cuda_ret = cudaDeviceSynchronize();
        err_check(cuda_ret, (char*)"Unable to launch particle motion kernel!", 5);
        iterations++;
    }
    // Launch the paper colouring kernel
    colouring_kernel <<< dimGrid, dimBlock >>> (
        device_particles,   // Particles allocated on device memory
        device_grid,        // paper grid allocated on device memory
        num_particles,      // number of particles
        GRID_SIZE,          // Grid Size
        BLEND);             // Blend ratio                

    // Synchronize threads and check for error 
    cuda_ret = cudaDeviceSynchronize();
    err_check(cuda_ret, (char*)"Unable to launch colouring kernel!", 6);

    // Stop clock
    clock_gettime(CLOCK_REALTIME, &end);

    // Copy the final particle positions back from the device to host
    cuda_ret = cudaMemcpy(particles, device_particles, num_particles * sizeof(AerosolCan), cudaMemcpyDeviceToHost);
    err_check(cuda_ret, (char*)"Unable to read particles from device memory!", 7);

    // Copy the final paper grid from the device to host
    cuda_ret = cudaMemcpy(grid, device_grid, GRID_SIZE * GRID_SIZE * sizeof(Colour), cudaMemcpyDeviceToHost);
    err_check(cuda_ret, (char*)"Unable to read paper grid from device memory!", 8);

    // Calculate elapsed time
    double ElapsedTime = (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / BILLION;

    // Write pixels data to output file
    FILE* fptr = fopen("Output.txt", "w");
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            fprintf(fptr, "%f  ", grid[i * GRID_SIZE + j].R);
            fprintf(fptr, "%f  ", grid[i * GRID_SIZE + j].G);
            fprintf(fptr, "%f\n", grid[i * GRID_SIZE + j].B);
        }
    }
    fclose(fptr);

    // Write final particle positions to output file
    FILE* fptr1 = fopen("FinalPosition.txt", "w");
    fprintf(fptr1, "%.2f  ", radius);
    int num_particles_hit = 0;
    for (int i = 0; i < num_particles; i++)
    {
        if (particles[i].check_hit == 1)
            num_particles_hit++;
        fprintf(fptr1, "%f  ", particles[i].pos.x);
        fprintf(fptr1, "%f  ", particles[i].pos.y);
        fprintf(fptr1, "%f\n", particles[i].pos.z);
    }
    fclose(fptr1);

    // Print the execution time
    printf("GPU Kernel Execution Time:  %lf seconds\n", ElapsedTime);
    printf("Total Number of Particles:  %d\n", num_particles);
    printf("Particles that hit the paper:  %d\n\n", num_particles_hit);

    return 0;
}

/* Error Check ----------------- //
*   Exits if there is a CUDA error.
*/
void err_check(cudaError_t ret, char* msg, int exit_code) {
    if (ret != cudaSuccess)
        fprintf(stderr, "%s \"%s\".\n", msg, cudaGetErrorString(ret)),
        exit(exit_code);
} // End Error Check ----------- //