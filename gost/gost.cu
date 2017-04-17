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

