// aerosol_can.h
#ifndef AEROSOL_CAN_H
#define AEROSOL_CAN_H

/* Structure for storing RGB values of pixel colour */
typedef struct
{
    double R;
    double G;
    double B;
} Colour;

/* Structure for storing position of a particle*/
typedef struct
{
    double x;           // x coordinate
    double y;           // y coordinate
    double z;           // z coordinate
} Vector;

/* Structure for storing information of a particle sprayed from Aerosol Can */
typedef struct
{
    Colour col;         // Color of Aerosol Can particles
    Vector pos;         // Position of Aerosol Can's nose
    Vector vel;         // Velocity of particle in the x, y and z direction
    Vector acc;         // Acceleration of particle in the x, y and z, direction
    int check_hit;      // For checking whether the paint particle hits the sheet
} AerosolCan;

#endif // AEROSOL_CAN_H
