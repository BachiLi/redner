#include "sobol_sampler.h"
#include "parallel.h"

#include "sobol.inc"

// // Initialize Sobol sampler's scramble parameter for each pixel
// struct sobol_initializer {
//     // Use PCG to initialize
//     // http://www.pcg-random.org/download.html
//     DEVICE inline uint32_t next_pcg32(uint64_t &state, uint64_t inc) {
//         uint64_t oldstate = state;
//         // Advance internal state
//         state = oldstate * 6364136223846793005ULL + (inc|1);
//         // Calculate output function (XSH RR), uses old state for max ILP
//         uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
//         uint32_t rot = oldstate >> 59u;
//         return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
//     }

//     DEVICE void operator()(int idx) {
//         // XXX: A little hacky. Should probably come up with my own scrambler
//         uint64_t state = 0U;
//         uint64_t inc = (((uint64_t)idx + 1) << 1u) | 1u;
//         next_pcg32(state, inc);
//         state += (0x853c49e6748fea9bULL + seed);
//         sobol_scramble[idx] = next_pcg32(state, inc);
//     }

//     uint64_t seed;
//     uint64_t *sobol_scramble;
// };

uint64_t hash64shift(uint64_t key) {
  key = (~key) + (key << 21); // key = (key << 21) - key - 1;
  key = key ^ (key >> 24);
  key = (key + (key << 3)) + (key << 8); // key * 265
  key = key ^ (key >> 14);
  key = (key + (key << 2)) + (key << 4); // key * 21
  key = key ^ (key >> 28);
  key = key + (key << 31);
  return key;
}

const uint64_t *sobol_matrices_cpu = &sobol::matrices_[0];
const uint64_t *sobol_matrices_gpu = nullptr;

SobolSampler::SobolSampler(bool use_gpu, uint64_t seed, int num_pixels) :
        use_gpu(use_gpu), current_dimension(0) {
    // sobol_scramble = Buffer<uint64_t>(use_gpu, num_pixels);
    // parallel_for(sobol_initializer{seed, sobol_scramble.begin()},
    //     sobol_scramble.size(), use_gpu);
    sobol_scramble = hash64shift(seed);
    sobol_matrices = use_gpu ? sobol_matrices_gpu : sobol_matrices_cpu;
    if (use_gpu && sobol_matrices == nullptr) {
#ifdef __CUDACC__
        checkCuda(cudaMallocManaged(&sobol_matrices_gpu, sizeof(sobol::matrices_)));
        checkCuda(cudaMemcpy((void*)sobol_matrices_gpu,
                             (void*)sobol::matrices_, sizeof(sobol::matrices_), cudaMemcpyHostToDevice));
        sobol_matrices = sobol_matrices_gpu;
#else
        assert(false);
#endif 
    }
}

// From Leonhard Gruenschloss (http://gruenschloss.org)
// Compute one component of the Sobol'-sequence, where the component
// corresponds to the dimension parameter, and the index specifies
// the point inside the sequence. The scramble parameter can be used
// to permute elementary intervals, and might be chosen randomly to
// generate a randomized QMC sequence. Only the Matrices::size least
// significant bits of the scramble value are used.
DEVICE
inline double sample(const uint64_t *matrices,
                     uint64_t index,
                     const uint32_t dimension,
                     const uint64_t scramble = 0ULL) {
    uint64_t result = scramble & ~-(1ULL << sobol::size);
    for (uint32_t i = dimension * sobol::size; index; index >>= 1, ++i) {
        if (index & 1) {
            result ^= matrices[i];
        }
    }

    double ret = result * (1.0 / (1ULL << sobol::size));
    return ret;
}

template <int spp, typename T>
struct sobol_sampler {
    DEVICE void operator()(int idx) {
        for (int i = 0; i < spp; i++) {
            samples[spp * idx + i] =
                (T)sample(sobol_matrices, idx, current_dimension + i, sobol_scramble);
        }
    }

    int current_dimension;
    const uint64_t *sobol_matrices;
    uint64_t sobol_scramble;
    T *samples;
};

void SobolSampler::next_camera_samples(BufferView<TCameraSample<float>> samples) {
    parallel_for(sobol_sampler<2, float>{
        current_dimension,
        sobol_matrices,
        sobol_scramble,
        (float*)samples.begin()}, samples.size(), use_gpu);
    current_dimension += 2;
}

void SobolSampler::next_camera_samples(BufferView<TCameraSample<double>> samples) {
    parallel_for(sobol_sampler<2, double>{
        current_dimension,
        sobol_matrices,
        sobol_scramble,
        (double*)samples.begin()}, samples.size(), use_gpu);
    current_dimension += 2;
}

void SobolSampler::next_light_samples(BufferView<TLightSample<float>> samples) {
    parallel_for(sobol_sampler<4, float>{
        current_dimension,
        sobol_matrices,
        sobol_scramble,
        (float*)samples.begin()}, samples.size(), use_gpu);
    current_dimension += 4;
}

void SobolSampler::next_light_samples(BufferView<TLightSample<double>> samples) {
    parallel_for(sobol_sampler<4, double>{
        current_dimension,
        sobol_matrices,
        sobol_scramble,
        (double*)samples.begin()}, samples.size(), use_gpu);
    current_dimension += 4;
}

void SobolSampler::next_bsdf_samples(BufferView<TBSDFSample<float>> samples) {
    parallel_for(sobol_sampler<3, float>{
        current_dimension,
        sobol_matrices,
        sobol_scramble,
        (float*)samples.begin()}, samples.size(), use_gpu);
    current_dimension += 3;
}

void SobolSampler::next_bsdf_samples(BufferView<TBSDFSample<double>> samples) {
    parallel_for(sobol_sampler<3, double>{
        current_dimension,
        sobol_matrices,
        sobol_scramble,
        (double*)samples.begin()}, samples.size(), use_gpu);
    current_dimension += 3;
}

void SobolSampler::next_primary_edge_samples(
        BufferView<TPrimaryEdgeSample<float>> samples) {
    parallel_for(sobol_sampler<2, float>{
        current_dimension,
        sobol_matrices,
        sobol_scramble,
        (float*)samples.begin()}, samples.size(), use_gpu);
    current_dimension += 2;
}

void SobolSampler::next_primary_edge_samples(
        BufferView<TPrimaryEdgeSample<double>> samples) {
    parallel_for(sobol_sampler<2, double>{
        current_dimension,
        sobol_matrices,
        sobol_scramble,
        (double*)samples.begin()}, samples.size(), use_gpu);
    current_dimension += 2;
}

void SobolSampler::next_secondary_edge_samples(
        BufferView<TSecondaryEdgeSample<float>> samples) {
    parallel_for(sobol_sampler<4, float>{
        current_dimension,
        sobol_matrices,
        sobol_scramble,
        (float*)samples.begin()}, samples.size(), use_gpu);
    current_dimension += 4;
}

void SobolSampler::next_secondary_edge_samples(
        BufferView<TSecondaryEdgeSample<double>> samples) {
    parallel_for(sobol_sampler<4, double>{
        current_dimension,
        sobol_matrices,
        sobol_scramble,
        (double*)samples.begin()}, samples.size(), use_gpu);
    current_dimension += 4;
}
