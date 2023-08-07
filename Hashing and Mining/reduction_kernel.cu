
/* Reduction Kernel ----------------------------------
*       Finds the minimum hash and nonce on the gpu
*/
__global__
void reduction_kernel(unsigned int* hash_array, unsigned int* nonce_array, int trials, unsigned int mod, unsigned int* min_hash, unsigned int* min_nonce) {

    // Calculate thread id
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    unsigned int local_min_hash = mod;
    unsigned int local_min_nonce = mod;

    for (int i = tid; i < trials; i += blockDim.x * gridDim.x) {
        if (hash_array[i] < local_min_hash) {
            local_min_hash = hash_array[i];
            local_min_nonce = nonce_array[i];
        }
    }

    atomicMin(min_hash, local_min_hash);
    if (local_min_hash == *min_hash) {
        atomicMin(min_nonce, local_min_nonce);
    }
    

} // End reduction Kernel //