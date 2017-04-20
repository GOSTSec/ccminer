extern "C" {
#include "sph/sph_streebog.h"
}

#include "miner.h"
#include "cuda_helper.h"

#include <stdio.h>
#include <memory.h>

#define NBN 2
static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

// GOST CPU Hash
extern "C" void gosthash(void *output, const void *input)
{
	unsigned char _ALIGN(128) hash[128] = { 0 };

	sph_gost512_context ctx_gost1;
	sph_gost256_context ctx_gost2;	

	sph_gost512_init(&ctx_gost1);
	sph_gost512(&ctx_gost1, (const void*) input, 80);
	sph_gost512_close(&ctx_gost1, (void*) hash);

	sph_gost256_init(&ctx_gost2);
	sph_gost256(&ctx_gost2, (const void*)hash, 64);
	sph_gost256_close(&ctx_gost1, (void*) hash);

	memcpy(output, hash, 32);
}

extern void gost_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void gost_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);

//#define _DEBUG
#define _DEBUG_PREFIX "sib"
#include "cuda_debug.cuh"

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_gost(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] >= 500 && !is_windows()) ? 19 : 18; // 2^18 = 262144 cuda threads
	if (device_sm[dev_id] >= 600) intensity = 20;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0xf;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), -1);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	do {
		int order = 0;

		// Hash with CUDA
		gost_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id]);
		TRACE("gost64   :");
		gost_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id]);
		TRACE("gost32   :");

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			gosthash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[1] =cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				*hashes_done = pdata[19] - first_nonce + throughput;
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					sibhash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;

	return 0;
}

