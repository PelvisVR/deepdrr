#include <stdint.h>

#define MAX_INTERSECTIONS MAX_INTERSECTIONS_SUB

// Function to fix odd ray intersections.

// Method:
// Start with sorted list of ray intersections
// Loop through once forwards to generate the altitude before and after each intersection
// The highest of the two ends is "sea level"
// Loop through and delete any collision that either starts or ends below sea level
// To do this, set the intercept Ts to infinity, decrement the count, and set facing to 0
// Then loop through to fill gaps
// Keep a pointer at at the highest filled index, and another that sweeps forward palcing into the lowest index
__device__ void tide(
    int *interceptCounts,
    float *interceptTs,
    int8_t *interceptFacing,
    int rayIdx,
    float sourceToDetectorDistance
)
{
    {
        // for (int i = 0; i < MAX_INTERSECTIONS; i++) { // TODO
        //     interceptFacing[i] = i < MAX_INTERSECTIONS / 2 ? 1 : -1;
        // }
        for (int i = 0; i < MAX_INTERSECTIONS; i+=4) {
            interceptFacing[i] = 1;
            interceptFacing[i+1] = 1;
            interceptFacing[i+2] = -1;
            interceptFacing[i+3] = -1;
            interceptTs[i] = -interceptTs[i];
            interceptTs[i+1] = interceptTs[i+1];
            interceptTs[i+2] = -interceptTs[i+2];
            interceptTs[i+3] = interceptTs[i+3];
        }
    }

    {
        (*interceptCounts) = 0;
        float cutoffEpsilon = 0.00001;
        for (int i = 0; i < MAX_INTERSECTIONS; i++) {
            if (interceptTs[i] < cutoffEpsilon || interceptTs[i] > sourceToDetectorDistance-0.001) {
                interceptTs[i] = INFINITY;
                interceptFacing[i] = 0;
            } else {
                (*interceptCounts) += 1;
            }
        }
    }

    {
        // selection sort h_interceptTs
        int sortedIdx = 0;
        while (sortedIdx < MAX_INTERSECTIONS) {
            int minIdx = sortedIdx;
            float minT = interceptTs[minIdx];
            for (int i = sortedIdx + 1; i < MAX_INTERSECTIONS; i++) {
                float t = interceptTs[i];
                if (t < minT) {
                    minIdx = i;
                    minT = t;
                }
            }
            float tmpT = interceptTs[sortedIdx];
            interceptTs[sortedIdx] = minT;
            interceptTs[minIdx] = tmpT;
            int8_t tmpFacing = interceptFacing[sortedIdx];
            interceptFacing[sortedIdx] = interceptFacing[minIdx];
            interceptFacing[minIdx] = tmpFacing;
            sortedIdx++;
        }
    }

    // remove t duplicates
    {
        int dstIdx = 0;
        int srcIdx = 1;
        while (srcIdx < MAX_INTERSECTIONS) {
            if (interceptTs[srcIdx] == interceptTs[dstIdx]) {
                interceptTs[srcIdx] = INFINITY;
                interceptFacing[srcIdx] = 0;
                (*interceptCounts) -= 1;
                srcIdx++;
            } else {
                dstIdx = srcIdx;
                srcIdx++;
            }
        }
    }

    {
        // Fill gaps
        int dstIdx = 0;
        int srcIdx = 0;

        while (dstIdx < MAX_INTERSECTIONS && interceptFacing[dstIdx] != 0) {
            dstIdx++;
        }
        srcIdx = dstIdx + 1;

        while (srcIdx < MAX_INTERSECTIONS && dstIdx < MAX_INTERSECTIONS) {
            while (srcIdx < MAX_INTERSECTIONS && interceptFacing[srcIdx] == 0) {
                srcIdx++;
            }
            if (srcIdx < MAX_INTERSECTIONS) {
                interceptTs[dstIdx] = interceptTs[srcIdx];
                interceptFacing[dstIdx] = interceptFacing[srcIdx];
                interceptTs[srcIdx] = INFINITY;
                interceptFacing[srcIdx] = 0;
            }
            srcIdx++;
            dstIdx++;
        }
    }

    {
        
        int altitudes[64]; // TODO
        int altitude = 0;
        
        for (int i = 0; i < MAX_INTERSECTIONS; i++) {
            altitude += interceptFacing[i];
            altitudes[i] = altitude;
        }
    
        int seaLevel = max(0, altitude);
    
        int prevAltitide = 0;
        for (int i = 0; i < MAX_INTERSECTIONS; i++) {
            int currentAltitude = altitudes[i];
            if (currentAltitude < seaLevel || prevAltitide < seaLevel) {
                interceptTs[i] = INFINITY;
                (*interceptCounts) -= interceptFacing[i] != 0;
                interceptFacing[i] = 0;
            }
            prevAltitide = currentAltitude;
        }
    }
    {
        // Fill gaps
        int dstIdx = 0;
        int srcIdx = 0;

        while (dstIdx < MAX_INTERSECTIONS && interceptFacing[dstIdx] != 0) {
            dstIdx++;
        }
        srcIdx = dstIdx + 1;

        while (srcIdx < MAX_INTERSECTIONS && dstIdx < MAX_INTERSECTIONS) {
            while (srcIdx < MAX_INTERSECTIONS && interceptFacing[srcIdx] == 0) {
                srcIdx++;
            }
            if (srcIdx < MAX_INTERSECTIONS) {
                interceptTs[dstIdx] = interceptTs[srcIdx];
                interceptFacing[dstIdx] = interceptFacing[srcIdx];
                interceptTs[srcIdx] = INFINITY;
                interceptFacing[srcIdx] = 0;
            }
            srcIdx++;
            dstIdx++;
        }
    }
}

__global__ void kernelTide(
    int* __restrict__ rayInterceptCounts,
    float* __restrict__ rayInterceptTs,
    int8_t* __restrict__ rayInterceptFacing,
    // int* __restrict__ detected,
    // int numTriangles, 
    int numRays,
    float sourceToDetectorDistance
)
{
    __shared__ int stride;
    if (threadIdx.x == 0) {
        stride = gridDim.x * blockDim.x;
    }
    __syncthreads();

    int threadStartIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = threadStartIdx; idx < numRays; idx += stride) {
        if (idx < numRays) {
            int *interceptCounts = rayInterceptCounts + idx;
            float *interceptTs = rayInterceptTs + idx * MAX_INTERSECTIONS;
            int8_t *interceptFacing = rayInterceptFacing + idx * MAX_INTERSECTIONS;
            tide(interceptCounts, interceptTs, interceptFacing, idx, sourceToDetectorDistance);
        }
    }
}

__device__ void reorder(
    float* __restrict__ rayInterceptTsIn,
    float* __restrict__ rayInterceptTsOut,
    int numRays,
    int rayIdx
)
{
    int num_layers = 4;
    for (int i = 0; i < MAX_INTERSECTIONS / num_layers; i++) {
        for (int j = 0; j < num_layers; j++) {
            // rayInterceptTsOut[rayIdx * MAX_INTERSECTIONS + i] = rayInterceptTsIn[rayIdx * MAX_INTERSECTIONS + i];
            rayInterceptTsOut[rayIdx * MAX_INTERSECTIONS + i * num_layers + j] = rayInterceptTsIn[i * numRays * num_layers + rayIdx * num_layers + j];
        }
    }
}

__global__ void kernelReorder(
    float* __restrict__ rayInterceptTsIn,
    float* __restrict__ rayInterceptTsOut,
    int numRays
)
{
    __shared__ int stride;
    if (threadIdx.x == 0) {
        stride = gridDim.x * blockDim.x;
    }
    __syncthreads();

    int threadStartIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = threadStartIdx; idx < numRays; idx += stride) {
        if (idx < numRays) {
            reorder(rayInterceptTsIn, rayInterceptTsOut, numRays, idx);
        }
    }
}
    

