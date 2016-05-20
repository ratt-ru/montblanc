#include <montblanc/abstraction.cuh>

namespace montblanc {

// Ensure that the dimensions of the supplied block
// are not greater than (X,Y,Z)
dim3 shrink_small_dims(dim3 && block, int X, int Y, int Z)
{
    if(X < block.x)
        { block.x = X; }

    if(Y < block.y)
        { block.y = Y; }

    if(Z < block.z)
        { block.z = Z; }

    return std::move(block);
}

// Given our thread block size, calculate
// our grid
dim3 grid_from_thread_block(const dim3 & block, int X, int Y, int Z)
{
    int GX = (X + block.x - 1) / block.x;
    int GY = (Y + block.y - 1) / block.y;
    int GZ = (Z + block.z - 1) / block.z;

    return dim3(GX, GY, GZ);
}

} // namespace montblanc {