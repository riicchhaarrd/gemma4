/*
 * gemma4.c — Single-file C99 inference for Gemma 4 (GGUF quantized models)
 *
 * Supports: Q4_K, Q5_K, Q6_K, IQ1_M, BF16, F16, F32 quantization formats
 * Architecture: Gemma 4 with ISWA (Interleaved Sliding Window Attention),
 *               Per-Layer Embeddings (PLE), KV sharing, GeGLU FFN
 *
 * Build:  gcc -O2 -o gemma4 gemma4.c -lm -lpthread
 * Usage:  ./gemma4 model.gguf -p "Hello" -n 128 -t 0.7
 *
 * This is a POSIX implementation (uses mmap). Tested on Linux.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ── Constants ───────────────────────────────────────────────────────── */

#define GGUF_MAGIC	   0x46554747 /* "GGUF" little-endian */
#define QK_K		   256		  /* Super-block size for K-quants */
#define MAX_LAYERS	   64
#define MAX_KV_ENTRIES 8192
#define MAX_SEQ		   8192
#define MAX_TOKENS	   262144
#define IQ1S_DELTA	   0.125f

/* GGUF types */
enum
{
	GGUF_TYPE_U8 = 0,
	GGUF_TYPE_I8,
	GGUF_TYPE_U16,
	GGUF_TYPE_I16,
	GGUF_TYPE_U32,
	GGUF_TYPE_I32,
	GGUF_TYPE_F32,
	GGUF_TYPE_BOOL,
	GGUF_TYPE_STRING,
	GGUF_TYPE_ARRAY,
	GGUF_TYPE_U64,
	GGUF_TYPE_I64,
	GGUF_TYPE_F64
};

/* GGML tensor types */
enum
{
	GGML_F32 = 0,
	GGML_F16 = 1,
	GGML_Q4_0 = 2,
	GGML_Q4_1 = 3,
	GGML_Q5_0 = 6,
	GGML_Q5_1 = 7,
	GGML_Q8_0 = 8,
	GGML_Q8_1 = 9,
	GGML_Q2_K = 10,
	GGML_Q3_K = 11,
	GGML_Q4_K = 12,
	GGML_Q5_K = 13,
	GGML_Q6_K = 14,
	GGML_Q8_K = 15,
	GGML_IQ4_XS = 23,
	GGML_IQ1_M = 29,
	GGML_BF16 = 30
};

/* ── FP16 / BF16 conversion ─────────────────────────────────────────── */

static float fp16_to_f32(uint16_t h)
{
	uint32_t sign = (uint32_t)(h >> 15) << 31;
	uint32_t exp = (h >> 10) & 0x1f;
	uint32_t mant = h & 0x3ff;
	uint32_t f;
	if(exp == 0)
	{
		if(mant == 0)
		{
			f = sign;
		}
		else
		{
			exp = 1;
			while(!(mant & 0x400))
			{
				mant <<= 1;
				exp--;
			}
			mant &= 0x3ff;
			f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
		}
	}
	else if(exp == 31)
	{
		f = sign | 0x7f800000 | (mant << 13);
	}
	else
	{
		f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
	}
	float result;
	memcpy(&result, &f, 4);
	return result;
}

static float bf16_to_f32(uint16_t h)
{
	uint32_t f = (uint32_t)h << 16;
	float result;
	memcpy(&result, &f, 4);
	return result;
}

/* ── IQ1S grid lookup table (packed 2-bit, 2048 entries) ─────────────── */

static const uint16_t iq1s_grid_packed[2048] = {
	0x0000, 0x0002, 0x0005, 0x0008, 0x000a, 0x0011, 0x0015, 0x0020, 0x0022, 0x0028, 0x002a, 0x0045, 0x0051, 0x0054,
	0x0056, 0x0065, 0x0080, 0x0082, 0x0088, 0x008a, 0x0095, 0x00a0, 0x00a2, 0x00a8, 0x00aa, 0x0104, 0x0105, 0x0111,
	0x0114, 0x0116, 0x0119, 0x011a, 0x0125, 0x0141, 0x0146, 0x0149, 0x0152, 0x0155, 0x015a, 0x0161, 0x0164, 0x0166,
	0x0168, 0x0185, 0x0191, 0x0194, 0x0196, 0x01a5, 0x0200, 0x0202, 0x0208, 0x020a, 0x0215, 0x0220, 0x0222, 0x0228,
	0x022a, 0x0245, 0x0251, 0x0259, 0x0264, 0x0269, 0x0280, 0x0282, 0x0288, 0x028a, 0x0291, 0x0295, 0x0299, 0x02a0,
	0x02a2, 0x02a8, 0x02aa, 0x0411, 0x0414, 0x0416, 0x0425, 0x0441, 0x0449, 0x0455, 0x045a, 0x0464, 0x0465, 0x0491,
	0x0499, 0x04a5, 0x0501, 0x0504, 0x0505, 0x0506, 0x0515, 0x0518, 0x051a, 0x0529, 0x0540, 0x0545, 0x054a, 0x0550,
	0x0551, 0x0554, 0x0555, 0x0556, 0x0559, 0x0560, 0x0562, 0x0565, 0x0568, 0x056a, 0x0581, 0x0591, 0x0595, 0x0598,
	0x059a, 0x05a1, 0x05a4, 0x05a5, 0x05a6, 0x05a9, 0x0614, 0x0619, 0x0641, 0x0644, 0x0650, 0x0652, 0x0655, 0x0658,
	0x0660, 0x0661, 0x0666, 0x0669, 0x0685, 0x0691, 0x0694, 0x0699, 0x0800, 0x0802, 0x0808, 0x080a, 0x0815, 0x0820,
	0x0822, 0x0828, 0x082a, 0x0845, 0x0851, 0x0856, 0x0865, 0x0880, 0x0882, 0x0888, 0x088a, 0x0895, 0x08a0, 0x08a2,
	0x08a8, 0x08aa, 0x0905, 0x0911, 0x0914, 0x0919, 0x0924, 0x0925, 0x0941, 0x0950, 0x0951, 0x0955, 0x0961, 0x0964,
	0x0969, 0x0991, 0x0994, 0x0996, 0x0999, 0x09a5, 0x0a00, 0x0a02, 0x0a08, 0x0a0a, 0x0a15, 0x0a20, 0x0a22, 0x0a28,
	0x0a2a, 0x0a45, 0x0a51, 0x0a59, 0x0a61, 0x0a65, 0x0a80, 0x0a82, 0x0a85, 0x0a88, 0x0a8a, 0x0a95, 0x0aa0, 0x0aa2,
	0x0aa8, 0x0aaa, 0x1010, 0x1011, 0x1014, 0x1019, 0x1024, 0x1025, 0x1041, 0x1044, 0x1050, 0x1055, 0x1058, 0x1061,
	0x1064, 0x1065, 0x1069, 0x1091, 0x1094, 0x1096, 0x10a1, 0x10a5, 0x1101, 0x1104, 0x1106, 0x1109, 0x1110, 0x1112,
	0x1115, 0x1118, 0x1121, 0x1124, 0x1129, 0x1145, 0x114a, 0x1150, 0x1151, 0x1152, 0x1154, 0x1155, 0x1156, 0x1159,
	0x1160, 0x1165, 0x1184, 0x1192, 0x1195, 0x11a1, 0x11a4, 0x1211, 0x1214, 0x1216, 0x1225, 0x1240, 0x1246, 0x1249,
	0x1252, 0x1255, 0x1258, 0x125a, 0x1264, 0x1266, 0x1285, 0x1291, 0x1294, 0x1296, 0x12a5, 0x1401, 0x1406, 0x1409,
	0x1414, 0x1415, 0x1418, 0x1419, 0x1421, 0x1426, 0x1441, 0x1445, 0x1446, 0x1448, 0x144a, 0x1451, 0x1454, 0x1455,
	0x1456, 0x1459, 0x1462, 0x1465, 0x1468, 0x1484, 0x1489, 0x1490, 0x1494, 0x1495, 0x1498, 0x1499, 0x149a, 0x14a1,
	0x14a4, 0x14a5, 0x14a9, 0x1502, 0x1505, 0x150a, 0x1511, 0x1514, 0x1515, 0x1516, 0x1519, 0x1520, 0x1522, 0x1525,
	0x1528, 0x152a, 0x1541, 0x1544, 0x1545, 0x1546, 0x1551, 0x1552, 0x1554, 0x1555, 0x1556, 0x1559, 0x155a, 0x1561,
	0x1564, 0x1565, 0x1566, 0x1569, 0x1580, 0x1582, 0x1584, 0x1585, 0x1588, 0x158a, 0x1590, 0x1591, 0x1594, 0x1595,
	0x1596, 0x1599, 0x159a, 0x15a0, 0x15a2, 0x15a5, 0x1601, 0x1604, 0x1605, 0x1606, 0x1615, 0x1616, 0x1618, 0x161a,
	0x1621, 0x1626, 0x1640, 0x1642, 0x1644, 0x1645, 0x1648, 0x164a, 0x1651, 0x1655, 0x1656, 0x1658, 0x1659, 0x1661,
	0x1664, 0x1665, 0x1668, 0x1669, 0x166a, 0x1686, 0x168a, 0x1692, 0x1695, 0x16a4, 0x16a9, 0x1811, 0x1816, 0x1825,
	0x1841, 0x1844, 0x1846, 0x1849, 0x1850, 0x1855, 0x1858, 0x185a, 0x1860, 0x1861, 0x1864, 0x1866, 0x1869, 0x1885,
	0x1891, 0x1894, 0x18a5, 0x1910, 0x1912, 0x1915, 0x191a, 0x1921, 0x1925, 0x1942, 0x1944, 0x1945, 0x1948, 0x1951,
	0x1954, 0x1955, 0x1956, 0x1959, 0x195a, 0x1960, 0x1965, 0x196a, 0x1989, 0x1991, 0x1992, 0x1995, 0x1998, 0x19a1,
	0x19a6, 0x19a9, 0x1a09, 0x1a16, 0x1a24, 0x1a26, 0x1a44, 0x1a46, 0x1a49, 0x1a50, 0x1a52, 0x1a55, 0x1a58, 0x1a61,
	0x1a66, 0x1a69, 0x1a85, 0x1a91, 0x1a96, 0x1a9a, 0x2000, 0x2002, 0x2008, 0x200a, 0x2015, 0x2020, 0x2022, 0x2025,
	0x2028, 0x202a, 0x2045, 0x2051, 0x2059, 0x2061, 0x2065, 0x2080, 0x2082, 0x2088, 0x208a, 0x2095, 0x20a0, 0x20a2,
	0x20a5, 0x20a8, 0x20aa, 0x2105, 0x2111, 0x2114, 0x2119, 0x2125, 0x2142, 0x2144, 0x2149, 0x2155, 0x2158, 0x215a,
	0x2161, 0x2164, 0x2165, 0x2166, 0x2185, 0x2190, 0x2196, 0x2199, 0x21a5, 0x2201, 0x2208, 0x220a, 0x2211, 0x2215,
	0x2220, 0x2222, 0x2228, 0x222a, 0x2245, 0x2251, 0x2256, 0x2259, 0x2265, 0x2281, 0x2288, 0x228a, 0x2291, 0x2295,
	0x22a0, 0x22a2, 0x22a8, 0x22aa, 0x2405, 0x2414, 0x2416, 0x2419, 0x2425, 0x2444, 0x2445, 0x2446, 0x2449, 0x2452,
	0x2455, 0x2458, 0x245a, 0x2466, 0x2485, 0x2491, 0x2494, 0x2499, 0x24a1, 0x24a5, 0x2509, 0x2515, 0x2521, 0x2529,
	0x2540, 0x2545, 0x2548, 0x2551, 0x2554, 0x2555, 0x2559, 0x2562, 0x2565, 0x2568, 0x2589, 0x2590, 0x2594, 0x2595,
	0x2598, 0x259a, 0x25a1, 0x25a4, 0x25a6, 0x25a9, 0x2605, 0x2610, 0x2612, 0x2619, 0x2625, 0x2641, 0x2649, 0x2655,
	0x2660, 0x2661, 0x2669, 0x2684, 0x2686, 0x2690, 0x269a, 0x2800, 0x2802, 0x2808, 0x280a, 0x2815, 0x2820, 0x2822,
	0x2828, 0x282a, 0x2845, 0x2851, 0x2854, 0x2865, 0x2880, 0x2882, 0x2888, 0x288a, 0x28a0, 0x28a2, 0x28a8, 0x28aa,
	0x2909, 0x2911, 0x2914, 0x2919, 0x2925, 0x2946, 0x2949, 0x2952, 0x2955, 0x2961, 0x2964, 0x2966, 0x2969, 0x2985,
	0x2990, 0x2996, 0x2999, 0x29a4, 0x29a5, 0x2a00, 0x2a02, 0x2a08, 0x2a0a, 0x2a20, 0x2a22, 0x2a28, 0x2a2a, 0x2a45,
	0x2a51, 0x2a56, 0x2a59, 0x2a65, 0x2a80, 0x2a82, 0x2a88, 0x2a8a, 0x2a95, 0x2aa0, 0x2aa2, 0x2aa8, 0x2aaa, 0x4005,
	0x4011, 0x4016, 0x4025, 0x4049, 0x4052, 0x4055, 0x4058, 0x405a, 0x4061, 0x4064, 0x4066, 0x4094, 0x4099, 0x40a1,
	0x40a6, 0x4100, 0x4101, 0x4104, 0x4106, 0x4109, 0x4112, 0x4115, 0x4116, 0x4118, 0x411a, 0x4121, 0x4126, 0x4129,
	0x4145, 0x4148, 0x414a, 0x4151, 0x4154, 0x4155, 0x4156, 0x4159, 0x415a, 0x4165, 0x4168, 0x416a, 0x4181, 0x4184,
	0x4186, 0x4190, 0x4192, 0x4195, 0x41a0, 0x41a1, 0x41a2, 0x4205, 0x4211, 0x4214, 0x4216, 0x4225, 0x4241, 0x4252,
	0x4255, 0x425a, 0x4264, 0x4269, 0x4289, 0x4294, 0x42a5, 0x4401, 0x4415, 0x4419, 0x4429, 0x4445, 0x4448, 0x444a,
	0x4451, 0x4454, 0x4455, 0x4456, 0x4461, 0x4462, 0x4465, 0x4468, 0x446a, 0x4481, 0x4486, 0x4489, 0x4490, 0x4492,
	0x4495, 0x44a0, 0x44a1, 0x44a9, 0x4501, 0x4502, 0x4505, 0x450a, 0x4511, 0x4514, 0x4515, 0x4516, 0x4519, 0x4520,
	0x4525, 0x452a, 0x4541, 0x4544, 0x4545, 0x4546, 0x4549, 0x4550, 0x4551, 0x4554, 0x4555, 0x4556, 0x4558, 0x4559,
	0x4561, 0x4564, 0x4565, 0x4566, 0x4569, 0x4582, 0x4584, 0x4585, 0x4588, 0x4591, 0x4594, 0x4595, 0x4596, 0x4599,
	0x459a, 0x45a5, 0x45a8, 0x45aa, 0x4601, 0x4605, 0x4609, 0x4614, 0x4615, 0x4618, 0x461a, 0x4621, 0x4624, 0x4629,
	0x4640, 0x4642, 0x4645, 0x4648, 0x4650, 0x4651, 0x4652, 0x4655, 0x4656, 0x4659, 0x4662, 0x4665, 0x4668, 0x4681,
	0x4685, 0x468a, 0x4694, 0x4695, 0x46a1, 0x46a4, 0x46a6, 0x4805, 0x4811, 0x4815, 0x481a, 0x4825, 0x4842, 0x4849,
	0x4850, 0x4855, 0x4858, 0x4861, 0x4864, 0x4866, 0x4869, 0x4885, 0x4891, 0x4894, 0x4896, 0x4899, 0x48a5, 0x4901,
	0x4905, 0x4906, 0x490a, 0x4910, 0x4914, 0x4915, 0x4918, 0x4921, 0x4924, 0x4926, 0x4940, 0x4945, 0x494a, 0x4951,
	0x4952, 0x4954, 0x4955, 0x4956, 0x4959, 0x4960, 0x4962, 0x4965, 0x4966, 0x496a, 0x4986, 0x4989, 0x4992, 0x4995,
	0x4996, 0x4998, 0x49a1, 0x49a4, 0x49a6, 0x49a9, 0x4a16, 0x4a44, 0x4a46, 0x4a49, 0x4a55, 0x4a58, 0x4a5a, 0x4a64,
	0x4a69, 0x4a94, 0x4aa5, 0x5001, 0x5004, 0x5005, 0x5006, 0x5009, 0x5012, 0x5015, 0x501a, 0x5021, 0x5024, 0x5029,
	0x5040, 0x5045, 0x5048, 0x5051, 0x5054, 0x5055, 0x5056, 0x5059, 0x5065, 0x5068, 0x5086, 0x5089, 0x5095, 0x5098,
	0x50a0, 0x50a1, 0x50a6, 0x50a9, 0x5105, 0x5108, 0x5109, 0x510a, 0x5111, 0x5114, 0x5115, 0x5116, 0x5118, 0x5119,
	0x5120, 0x5125, 0x5126, 0x5128, 0x512a, 0x5141, 0x5144, 0x5145, 0x5146, 0x5149, 0x5150, 0x5151, 0x5152, 0x5154,
	0x5155, 0x5156, 0x5158, 0x5159, 0x515a, 0x5161, 0x5164, 0x5165, 0x5166, 0x5169, 0x5182, 0x5185, 0x5191, 0x5194,
	0x5195, 0x5196, 0x5199, 0x51a0, 0x51a5, 0x51aa, 0x5201, 0x5206, 0x5212, 0x5215, 0x521a, 0x5221, 0x5224, 0x5242,
	0x5245, 0x524a, 0x5251, 0x5254, 0x5255, 0x5256, 0x5259, 0x5262, 0x5265, 0x5285, 0x5290, 0x5292, 0x5295, 0x5299,
	0x529a, 0x52a4, 0x5404, 0x5405, 0x5411, 0x5414, 0x5415, 0x5416, 0x5418, 0x5419, 0x5421, 0x5425, 0x5428, 0x542a,
	0x5441, 0x5444, 0x5445, 0x5446, 0x5449, 0x544a, 0x5450, 0x5451, 0x5454, 0x5455, 0x5456, 0x5458, 0x5459, 0x545a,
	0x5461, 0x5462, 0x5464, 0x5465, 0x5466, 0x5469, 0x5480, 0x5488, 0x548a, 0x5491, 0x5494, 0x5495, 0x5496, 0x5499,
	0x54a1, 0x54a4, 0x54a5, 0x54aa, 0x5501, 0x5502, 0x5504, 0x5505, 0x5506, 0x5509, 0x5510, 0x5511, 0x5512, 0x5514,
	0x5515, 0x5516, 0x5519, 0x551a, 0x5521, 0x5524, 0x5525, 0x5526, 0x5529, 0x5540, 0x5541, 0x5542, 0x5544, 0x5545,
	0x5546, 0x5548, 0x5549, 0x5550, 0x5551, 0x5552, 0x5554, 0x5555, 0x5556, 0x5558, 0x5559, 0x555a, 0x5560, 0x5561,
	0x5564, 0x5565, 0x5566, 0x5568, 0x5569, 0x556a, 0x5581, 0x5584, 0x5585, 0x5589, 0x558a, 0x5590, 0x5591, 0x5594,
	0x5595, 0x5596, 0x5598, 0x5599, 0x55a1, 0x55a4, 0x55a5, 0x55a6, 0x55a9, 0x5600, 0x5601, 0x5602, 0x5604, 0x5606,
	0x5608, 0x5609, 0x5611, 0x5614, 0x5615, 0x5618, 0x5619, 0x5620, 0x5621, 0x5622, 0x5624, 0x5625, 0x5626, 0x5628,
	0x5629, 0x5641, 0x5645, 0x5646, 0x5648, 0x5649, 0x564a, 0x5650, 0x5651, 0x5652, 0x5654, 0x5655, 0x5656, 0x5658,
	0x5659, 0x565a, 0x5661, 0x5664, 0x5665, 0x5669, 0x5682, 0x5685, 0x5686, 0x5688, 0x5689, 0x568a, 0x5691, 0x5695,
	0x569a, 0x56a2, 0x56a5, 0x56a6, 0x56a8, 0x56a9, 0x5804, 0x5805, 0x5806, 0x5809, 0x5810, 0x5815, 0x5818, 0x5821,
	0x582a, 0x5845, 0x5848, 0x584a, 0x5851, 0x5854, 0x5855, 0x5856, 0x5858, 0x5859, 0x5860, 0x5862, 0x5864, 0x5865,
	0x5882, 0x5889, 0x5890, 0x5892, 0x5895, 0x5898, 0x58a1, 0x58a9, 0x5901, 0x5902, 0x5905, 0x590a, 0x5911, 0x5914,
	0x5915, 0x5916, 0x5919, 0x5925, 0x5941, 0x5944, 0x5945, 0x5946, 0x5949, 0x5950, 0x5951, 0x5952, 0x5954, 0x5955,
	0x5956, 0x5958, 0x5959, 0x595a, 0x5961, 0x5964, 0x5965, 0x5966, 0x5969, 0x5981, 0x5985, 0x5989, 0x5991, 0x5994,
	0x5995, 0x5996, 0x5998, 0x5999, 0x59a5, 0x5a04, 0x5a08, 0x5a15, 0x5a1a, 0x5a20, 0x5a25, 0x5a26, 0x5a29, 0x5a45,
	0x5a48, 0x5a49, 0x5a51, 0x5a55, 0x5a56, 0x5a58, 0x5a59, 0x5a62, 0x5a65, 0x5a68, 0x5a6a, 0x5a81, 0x5a8a, 0x5a92,
	0x5a95, 0x5a96, 0x5a98, 0x5a9a, 0x5aa1, 0x6005, 0x6014, 0x6016, 0x6019, 0x6025, 0x6044, 0x6050, 0x6055, 0x6056,
	0x6058, 0x605a, 0x6061, 0x6064, 0x6066, 0x6069, 0x6081, 0x6096, 0x60a5, 0x6101, 0x6104, 0x6106, 0x6109, 0x6112,
	0x6115, 0x6121, 0x6122, 0x6126, 0x6129, 0x6145, 0x6149, 0x6151, 0x6155, 0x6156, 0x6159, 0x6165, 0x6166, 0x616a,
	0x6184, 0x618a, 0x6192, 0x6195, 0x61a1, 0x61a6, 0x61a9, 0x6211, 0x6216, 0x6219, 0x6240, 0x6241, 0x6246, 0x6255,
	0x6256, 0x6258, 0x6260, 0x6285, 0x6291, 0x6296, 0x62a5, 0x6411, 0x6412, 0x6415, 0x6416, 0x641a, 0x6421, 0x6426,
	0x6429, 0x6440, 0x6442, 0x6445, 0x6448, 0x644a, 0x6451, 0x6454, 0x6455, 0x6456, 0x6459, 0x645a, 0x6460, 0x6462,
	0x6465, 0x6484, 0x6485, 0x6489, 0x6490, 0x6492, 0x6494, 0x6495, 0x6496, 0x6498, 0x649a, 0x64a1, 0x64a4, 0x64a9,
	0x6505, 0x6508, 0x650a, 0x6511, 0x6515, 0x6516, 0x6519, 0x6544, 0x6545, 0x6546, 0x6549, 0x6550, 0x6551, 0x6554,
	0x6555, 0x6556, 0x6559, 0x6561, 0x6564, 0x6565, 0x6566, 0x6569, 0x6586, 0x6589, 0x658a, 0x6591, 0x6595, 0x6596,
	0x6599, 0x659a, 0x65a2, 0x65a5, 0x65a6, 0x65a8, 0x6602, 0x6609, 0x6615, 0x6620, 0x6626, 0x6628, 0x6629, 0x6640,
	0x6645, 0x6648, 0x664a, 0x6651, 0x6654, 0x6655, 0x6656, 0x6658, 0x665a, 0x6660, 0x6665, 0x6668, 0x6680, 0x6682,
	0x6685, 0x668a, 0x6694, 0x6696, 0x6698, 0x6699, 0x66a0, 0x66a4, 0x66a6, 0x66aa, 0x6816, 0x6819, 0x6825, 0x6841,
	0x6852, 0x6855, 0x685a, 0x6861, 0x6869, 0x6885, 0x6891, 0x6898, 0x68a6, 0x6901, 0x6904, 0x6910, 0x6915, 0x6921,
	0x6924, 0x6926, 0x6929, 0x6940, 0x6941, 0x6945, 0x6946, 0x6948, 0x6951, 0x6954, 0x6955, 0x6956, 0x6959, 0x6960,
	0x6965, 0x696a, 0x6982, 0x6984, 0x698a, 0x6995, 0x69a1, 0x69a4, 0x69a5, 0x69a9, 0x6a11, 0x6a16, 0x6a18, 0x6a41,
	0x6a44, 0x6a49, 0x6a50, 0x6a55, 0x6a58, 0x6a5a, 0x6a64, 0x6a65, 0x6a69, 0x6a86, 0x6a94, 0x6a98, 0x6a9a, 0x6aa6,
	0x8000, 0x8002, 0x8008, 0x800a, 0x8020, 0x8022, 0x8028, 0x802a, 0x8045, 0x8050, 0x8051, 0x8054, 0x8056, 0x8059,
	0x8065, 0x8080, 0x8082, 0x8088, 0x808a, 0x8095, 0x80a0, 0x80a2, 0x80a8, 0x80aa, 0x8105, 0x8111, 0x8114, 0x8116,
	0x8119, 0x8125, 0x8141, 0x8144, 0x8149, 0x8150, 0x8152, 0x8155, 0x8156, 0x8158, 0x8159, 0x8164, 0x8166, 0x8169,
	0x8185, 0x8189, 0x8194, 0x8196, 0x8199, 0x81a5, 0x8200, 0x8202, 0x8208, 0x820a, 0x8215, 0x8220, 0x8222, 0x8228,
	0x822a, 0x8251, 0x8254, 0x8259, 0x8265, 0x8280, 0x8282, 0x8288, 0x828a, 0x8295, 0x82a0, 0x82a2, 0x82a8, 0x82aa,
	0x8414, 0x8419, 0x8441, 0x8444, 0x8451, 0x8455, 0x845a, 0x8461, 0x8464, 0x8469, 0x8494, 0x8499, 0x8501, 0x8509,
	0x8512, 0x8515, 0x851a, 0x8526, 0x8529, 0x8540, 0x8541, 0x8545, 0x8548, 0x8551, 0x8554, 0x8555, 0x8556, 0x8559,
	0x855a, 0x8565, 0x8566, 0x8568, 0x856a, 0x8581, 0x8584, 0x8586, 0x8589, 0x8590, 0x8592, 0x8595, 0x8598, 0x85a6,
	0x8611, 0x8616, 0x8619, 0x8625, 0x8641, 0x8644, 0x8649, 0x864a, 0x8650, 0x8655, 0x8659, 0x865a, 0x8661, 0x8666,
	0x866a, 0x8685, 0x8691, 0x869a, 0x86a4, 0x8800, 0x8802, 0x8808, 0x880a, 0x8815, 0x8820, 0x8822, 0x8828, 0x882a,
	0x8841, 0x8845, 0x8851, 0x8854, 0x8859, 0x8865, 0x8869, 0x8880, 0x8882, 0x8888, 0x888a, 0x8895, 0x88a0, 0x88a2,
	0x88a8, 0x88aa, 0x8905, 0x8906, 0x8911, 0x8914, 0x8916, 0x8925, 0x8941, 0x8944, 0x8946, 0x8949, 0x8950, 0x8952,
	0x8955, 0x895a, 0x8961, 0x8964, 0x8985, 0x8996, 0x8999, 0x89a5, 0x8a00, 0x8a02, 0x8a08, 0x8a0a, 0x8a15, 0x8a20,
	0x8a22, 0x8a28, 0x8a2a, 0x8a45, 0x8a51, 0x8a54, 0x8a56, 0x8a80, 0x8a82, 0x8a88, 0x8a8a, 0x8a95, 0x8aa0, 0x8aa2,
	0x8aa8, 0x8aaa, 0x9005, 0x9011, 0x9016, 0x9018, 0x9019, 0x9025, 0x9041, 0x9046, 0x9049, 0x9055, 0x9058, 0x905a,
	0x9069, 0x906a, 0x9085, 0x9091, 0x9094, 0x9096, 0x9099, 0x90a5, 0x9101, 0x9104, 0x9106, 0x9109, 0x9110, 0x9115,
	0x9118, 0x911a, 0x9121, 0x9124, 0x9126, 0x9129, 0x9140, 0x9145, 0x9150, 0x9151, 0x9154, 0x9155, 0x9156, 0x9159,
	0x9162, 0x9165, 0x9184, 0x9186, 0x9192, 0x9195, 0x9198, 0x91a1, 0x91a4, 0x91a6, 0x91a9, 0x9205, 0x9211, 0x9214,
	0x9219, 0x9225, 0x9244, 0x9246, 0x9249, 0x9250, 0x9252, 0x9255, 0x9258, 0x9266, 0x9269, 0x9285, 0x9294, 0x9296,
	0x92a9, 0x9401, 0x9404, 0x9406, 0x9410, 0x9415, 0x9418, 0x9426, 0x9440, 0x944a, 0x9451, 0x9454, 0x9455, 0x9456,
	0x9458, 0x9459, 0x9460, 0x9461, 0x9462, 0x9465, 0x9484, 0x9486, 0x9492, 0x9494, 0x9495, 0x9498, 0x94a1, 0x94a9,
	0x9500, 0x9505, 0x9508, 0x950a, 0x9510, 0x9511, 0x9514, 0x9515, 0x9516, 0x9519, 0x9521, 0x9525, 0x9529, 0x952a,
	0x9541, 0x9544, 0x9545, 0x9546, 0x9549, 0x9550, 0x9551, 0x9552, 0x9554, 0x9555, 0x9556, 0x9558, 0x9559, 0x955a,
	0x9561, 0x9564, 0x9565, 0x9566, 0x9569, 0x9581, 0x9585, 0x9588, 0x9591, 0x9592, 0x9594, 0x9595, 0x9596, 0x9599,
	0x959a, 0x95a0, 0x95a2, 0x95a5, 0x95a8, 0x95aa, 0x9601, 0x9604, 0x9610, 0x9615, 0x9619, 0x9620, 0x9626, 0x9629,
	0x9645, 0x9648, 0x9649, 0x9651, 0x9652, 0x9655, 0x9656, 0x9659, 0x9665, 0x9668, 0x9682, 0x9684, 0x9689, 0x968a,
	0x9692, 0x9694, 0x9695, 0x96a4, 0x96a6, 0x96a9, 0x9805, 0x9816, 0x9819, 0x9825, 0x9841, 0x9846, 0x9850, 0x9852,
	0x9855, 0x9856, 0x985a, 0x9864, 0x9865, 0x9885, 0x9891, 0x9896, 0x9899, 0x98a5, 0x9904, 0x9906, 0x9909, 0x9910,
	0x9912, 0x9915, 0x9918, 0x991a, 0x9920, 0x9921, 0x9924, 0x9926, 0x9940, 0x9942, 0x9945, 0x9948, 0x994a, 0x9951,
	0x9954, 0x9955, 0x9956, 0x9959, 0x9962, 0x9965, 0x9966, 0x996a, 0x9981, 0x9984, 0x9990, 0x9992, 0x9995, 0x999a,
	0x99a1, 0x99a6, 0x9a05, 0x9a15, 0x9a25, 0x9a44, 0x9a46, 0x9a49, 0x9a50, 0x9a55, 0x9a58, 0x9a61, 0x9a85, 0x9a91,
	0x9a94, 0x9a95, 0x9a96, 0xa000, 0xa002, 0xa008, 0xa00a, 0xa015, 0xa020, 0xa022, 0xa028, 0xa02a, 0xa045, 0xa051,
	0xa054, 0xa056, 0xa059, 0xa080, 0xa082, 0xa088, 0xa08a, 0xa095, 0xa0a0, 0xa0a2, 0xa0a8, 0xa0aa, 0xa105, 0xa109,
	0xa111, 0xa114, 0xa116, 0xa119, 0xa11a, 0xa146, 0xa149, 0xa151, 0xa155, 0xa158, 0xa15a, 0xa161, 0xa164, 0xa185,
	0xa190, 0xa192, 0xa196, 0xa199, 0xa202, 0xa208, 0xa20a, 0xa210, 0xa219, 0xa222, 0xa228, 0xa22a, 0xa245, 0xa251,
	0xa256, 0xa259, 0xa265, 0xa280, 0xa282, 0xa288, 0xa28a, 0xa295, 0xa2a0, 0xa2a2, 0xa2a8, 0xa2aa, 0xa419, 0xa425,
	0xa441, 0xa444, 0xa450, 0xa454, 0xa455, 0xa458, 0xa45a, 0xa461, 0xa465, 0xa466, 0xa468, 0xa469, 0xa485, 0xa506,
	0xa509, 0xa510, 0xa512, 0xa515, 0xa518, 0xa526, 0xa529, 0xa542, 0xa545, 0xa551, 0xa554, 0xa555, 0xa556, 0xa559,
	0xa565, 0xa56a, 0xa581, 0xa584, 0xa585, 0xa586, 0xa589, 0xa592, 0xa595, 0xa598, 0xa605, 0xa611, 0xa616, 0xa61a,
	0xa621, 0xa625, 0xa644, 0xa646, 0xa64a, 0xa652, 0xa655, 0xa656, 0xa658, 0xa660, 0xa662, 0xa686, 0xa690, 0xa695,
	0xa696, 0xa699, 0xa6a1, 0xa6a4, 0xa6a6, 0xa800, 0xa802, 0xa808, 0xa80a, 0xa820, 0xa822, 0xa828, 0xa82a, 0xa851,
	0xa854, 0xa856, 0xa859, 0xa880, 0xa882, 0xa888, 0xa88a, 0xa895, 0xa8a0, 0xa8a2, 0xa8a8, 0xa8aa, 0xa905, 0xa914,
	0xa919, 0xa921, 0xa925, 0xa941, 0xa950, 0xa955, 0xa95a, 0xa961, 0xa966, 0xa969, 0xa990, 0xa996, 0xaa00, 0xaa02,
	0xaa08, 0xaa0a, 0xaa20, 0xaa22, 0xaa28, 0xaa2a, 0xaa51, 0xaa54, 0xaa56, 0xaa80, 0xaa82, 0xaa88, 0xaa8a, 0xaa95,
	0xaaa0, 0xaaa2, 0xaaa8, 0xaaaa,
};

static uint64_t iq1s_grid[2048]; /* Unpacked at startup from iq1s_grid_packed */

static void init_iq1s_grid(void)
{
	for(int i = 0; i < 2048; i++)
	{
		uint8_t bytes[8];
		uint16_t p = iq1s_grid_packed[i];
		for(int j = 0; j < 8; j++)
			bytes[j] = (uint8_t)((int8_t)(((p >> (j * 2)) & 3) - 1));
		memcpy(&iq1s_grid[i], bytes, 8);
	}
}

/* ── Quantization block structures ───────────────────────────────────── */

#pragma pack(push, 1)
typedef struct
{
	uint16_t d;
	uint16_t dmin;
	uint8_t scales[12];
	uint8_t qs[QK_K / 2];
} block_q4_K; /* 144 bytes */
typedef struct
{
	uint16_t d;
	uint16_t dmin;
	uint8_t scales[12];
	uint8_t qh[QK_K / 8];
	uint8_t qs[QK_K / 2];
} block_q5_K; /* 176 bytes */
typedef struct
{
	uint8_t ql[QK_K / 2];
	uint8_t qh[QK_K / 4];
	int8_t scales[QK_K / 16];
	uint16_t d;
} block_q6_K; /* 210 bytes */
typedef struct
{
	uint8_t qs[QK_K / 8];
	uint8_t qh[QK_K / 16];
	uint8_t scales[QK_K / 32];
} block_iq1_m; /* 56 bytes */
typedef struct
{
	uint16_t d;
	uint16_t scales_h;
	uint8_t scales_l[QK_K / 64];
	uint8_t qs[QK_K / 2];
} block_iq4_xs; /* 136 bytes */
#pragma pack(pop)

/* IQ4_NL non-linear quantization grid */
static const int8_t kvalues_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

/* ── Dequantization ──────────────────────────────────────────────────── */

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
	if(j < 4)
	{
		*d = q[j] & 63;
		*m = q[j + 4] & 63;
	}
	else
	{
		*d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
		*m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
	}
}

static void dequantize_row_q4_k(const void *src, float *dst, int n)
{
	const block_q4_K *x = (const block_q4_K *)src;
	int nb = n / QK_K;
	for(int i = 0; i < nb; i++)
	{
		const uint8_t *q = x[i].qs;
		const float d = fp16_to_f32(x[i].d);
		const float min = fp16_to_f32(x[i].dmin);
		int is = 0;
		for(int j = 0; j < QK_K; j += 64)
		{
			uint8_t sc, m;
			get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
			float d1 = d * sc, m1 = min * m;
			get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
			float d2 = d * sc, m2 = min * m;
			for(int l = 0; l < 32; ++l)
				*dst++ = d1 * (q[l] & 0xF) - m1;
			for(int l = 0; l < 32; ++l)
				*dst++ = d2 * (q[l] >> 4) - m2;
			q += 32;
			is += 2;
		}
	}
}

static void dequantize_row_q5_k(const void *src, float *dst, int n)
{
	const block_q5_K *x = (const block_q5_K *)src;
	int nb = n / QK_K;
	for(int i = 0; i < nb; i++)
	{
		const uint8_t *ql = x[i].qs;
		const uint8_t *qh = x[i].qh;
		const float d = fp16_to_f32(x[i].d);
		const float min = fp16_to_f32(x[i].dmin);
		int is = 0;
		uint8_t u1 = 1, u2 = 2;
		for(int j = 0; j < QK_K; j += 64)
		{
			uint8_t sc, m;
			get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
			float d1 = d * sc, m1 = min * m;
			get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
			float d2 = d * sc, m2 = min * m;
			for(int l = 0; l < 32; ++l)
				*dst++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
			for(int l = 0; l < 32; ++l)
				*dst++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
			ql += 32;
			is += 2;
			u1 <<= 2;
			u2 <<= 2;
		}
	}
}

static void dequantize_row_q6_k(const void *src, float *dst, int n)
{
	const block_q6_K *x = (const block_q6_K *)src;
	int nb = n / QK_K;
	for(int i = 0; i < nb; i++)
	{
		const float d = fp16_to_f32(x[i].d);
		const uint8_t *ql = x[i].ql;
		const uint8_t *qh = x[i].qh;
		const int8_t *sc = x[i].scales;
		for(int n2 = 0; n2 < QK_K; n2 += 128)
		{
			for(int l = 0; l < 32; ++l)
			{
				int is = l / 16;
				int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
				int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
				int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
				int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
				dst[l + 0] = d * sc[is + 0] * q1;
				dst[l + 32] = d * sc[is + 2] * q2;
				dst[l + 64] = d * sc[is + 4] * q3;
				dst[l + 96] = d * sc[is + 6] * q4;
			}
			dst += 128;
			ql += 64;
			qh += 32;
			sc += 8;
		}
	}
}

static void dequantize_row_iq1_m(const void *src, float *dst, int n)
{
	const block_iq1_m *x = (const block_iq1_m *)src;
	int nb = n / QK_K;
	for(int i = 0; i < nb; i++)
	{
		const uint16_t *sc = (const uint16_t *)x[i].scales;
		/* Extract d from high 4 bits of each of the 4 scale uint16s */
		uint16_t d_bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
		float d = fp16_to_f32(d_bits);

		const uint8_t *qs = x[i].qs;
		const uint8_t *qh = x[i].qh;

		for(int ib = 0; ib < QK_K / 32; ++ib)
		{
			float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
			float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
			uint16_t idx[4];
			float delta[4];
			idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
			idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
			idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
			idx[3] = qs[3] | ((qh[1] << 4) & 0x700);
			delta[0] = (qh[0] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
			delta[1] = (qh[0] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
			delta[2] = (qh[1] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
			delta[3] = (qh[1] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
			for(int l = 0; l < 2; ++l)
			{
				const int8_t *grid = (const int8_t *)(iq1s_grid + idx[l]);
				for(int j = 0; j < 8; ++j)
					dst[j] = dl1 * (grid[j] + delta[l]);
				dst += 8;
			}
			for(int l = 2; l < 4; ++l)
			{
				const int8_t *grid = (const int8_t *)(iq1s_grid + idx[l]);
				for(int j = 0; j < 8; ++j)
					dst[j] = dl2 * (grid[j] + delta[l]);
				dst += 8;
			}
			qs += 4;
			qh += 2;
		}
	}
}

static void dequantize_row_iq4_xs(const void *src, float *dst, int n)
{
	const block_iq4_xs *x = (const block_iq4_xs *)src;
	int nb = n / QK_K;
	for(int i = 0; i < nb; i++)
	{
		const uint8_t *qs = x[i].qs;
		const float d = fp16_to_f32(x[i].d);
		for(int ib = 0; ib < QK_K / 32; ++ib)
		{
			int ls = ((x[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x[i].scales_h >> 2 * ib) & 3) << 4);
			float dl = d * (ls - 32);
			for(int j = 0; j < 16; ++j)
			{
				dst[j + 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
				dst[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
			}
			dst += 32;
			qs += 16;
		}
	}
}

static void dequantize_row(const void *src, float *dst, int n, int type)
{
	switch(type)
	{
		case GGML_F32: memcpy(dst, src, n * sizeof(float)); break;
		case GGML_F16:
			for(int i = 0; i < n; i++)
				dst[i] = fp16_to_f32(((const uint16_t *)src)[i]);
			break;
		case GGML_BF16:
			for(int i = 0; i < n; i++)
				dst[i] = bf16_to_f32(((const uint16_t *)src)[i]);
			break;
		case GGML_Q4_K: dequantize_row_q4_k(src, dst, n); break;
		case GGML_Q5_K: dequantize_row_q5_k(src, dst, n); break;
		case GGML_Q6_K: dequantize_row_q6_k(src, dst, n); break;
		case GGML_IQ1_M: dequantize_row_iq1_m(src, dst, n); break;
		case GGML_IQ4_XS: dequantize_row_iq4_xs(src, dst, n); break;
		default: fprintf(stderr, "Unsupported quant type: %d\n", type); exit(1);
	}
    for(int i = 0; i < n; ++i)
    printf("%f ", dst[i]);
    putchar('\n');
}

static size_t type_block_size(int type)
{
	switch(type)
	{
		case GGML_F32: return 4;
		case GGML_F16: return 2;
		case GGML_BF16: return 2;
		case GGML_Q4_K: return sizeof(block_q4_K);
		case GGML_Q5_K: return sizeof(block_q5_K);
		case GGML_Q6_K: return sizeof(block_q6_K);
		case GGML_IQ1_M: return sizeof(block_iq1_m);
		case GGML_IQ4_XS: return sizeof(block_iq4_xs);
		default: return 0;
	}
}

static int type_elements_per_block(int type)
{
	switch(type)
	{
		case GGML_F32:
		case GGML_F16:
		case GGML_BF16: return 1;
		default: return QK_K;
	}
}

/* Bytes per row of n elements */
static size_t row_bytes(int n, int type)
{
	int epb = type_elements_per_block(type);
	return (size_t)(n / epb) * type_block_size(type);
}

/* ── GGUF parser ─────────────────────────────────────────────────────── */

typedef struct
{
	const uint8_t *base;
	size_t pos;
	size_t size;
} Reader;

static uint32_t rd_u32(Reader *r)
{
	uint32_t v;
	memcpy(&v, r->base + r->pos, 4);
	r->pos += 4;
	return v;
}
static uint64_t rd_u64(Reader *r)
{
	uint64_t v;
	memcpy(&v, r->base + r->pos, 8);
	r->pos += 8;
	return v;
}
static float rd_f32(Reader *r)
{
	float v;
	memcpy(&v, r->base + r->pos, 4);
	r->pos += 4;
	return v;
}
static int32_t rd_i32(Reader *r)
{
	int32_t v;
	memcpy(&v, r->base + r->pos, 4);
	r->pos += 4;
	return v;
}

static const char *rd_str(Reader *r)
{
	uint64_t len = rd_u64(r);
	const char *s = (const char *)(r->base + r->pos);
	r->pos += len;
	return s;
}

static char *rd_str_dup(Reader *r)
{
	uint64_t len = rd_u64(r);
	char *s = malloc(len + 1);
	memcpy(s, r->base + r->pos, len);
	s[len] = 0;
	r->pos += len;
	return s;
}

/* Skip a GGUF value of given type */
static void skip_value(Reader *r, int type);

static size_t gguf_type_size(int type)
{
	switch(type)
	{
		case GGUF_TYPE_U8:
		case GGUF_TYPE_I8:
		case GGUF_TYPE_BOOL: return 1;
		case GGUF_TYPE_U16:
		case GGUF_TYPE_I16: return 2;
		case GGUF_TYPE_U32:
		case GGUF_TYPE_I32:
		case GGUF_TYPE_F32: return 4;
		case GGUF_TYPE_U64:
		case GGUF_TYPE_I64:
		case GGUF_TYPE_F64: return 8;
		default: return 0;
	}
}

static void skip_value(Reader *r, int type)
{
	if(type == GGUF_TYPE_STRING)
	{
		uint64_t len = rd_u64(r);
		r->pos += len;
	}
	else if(type == GGUF_TYPE_ARRAY)
	{
		uint32_t elem_type = rd_u32(r);
		uint64_t count = rd_u64(r);
		if(elem_type == GGUF_TYPE_STRING)
		{
			for(uint64_t i = 0; i < count; i++)
				skip_value(r, GGUF_TYPE_STRING);
		}
		else
		{
			r->pos += count * gguf_type_size(elem_type);
		}
	}
	else
	{
		r->pos += gguf_type_size(type);
	}
}

/* Tensor info as parsed from GGUF */
typedef struct
{
	char name[128];
	int n_dims;
	int64_t dims[4];
	int type;
	size_t offset; /* relative to tensor data start */
} TensorInfo;

typedef struct
{
	int fd;
	void *map;
	size_t map_size;
	uint32_t version;
	uint64_t n_tensors;
	uint64_t n_kv;
	TensorInfo *tensors;
	const uint8_t *data_start; /* Start of tensor data in mmap */

	/* Metadata storage — we parse on demand with key lookup */
	const uint8_t *kv_section; /* Start of KV pairs */
	size_t kv_section_size;
} GGUFFile;

/* Open and parse GGUF header + tensor info */
static GGUFFile *gguf_open(const char *path)
{
	int fd = open(path, O_RDONLY);
	if(fd < 0)
	{
		perror("open");
		exit(1);
	}

	struct stat st;
	fstat(fd, &st);
	size_t file_size = st.st_size;

	void *map = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if(map == MAP_FAILED)
	{
		perror("mmap");
		exit(1);
	}

	GGUFFile *g = calloc(1, sizeof(GGUFFile));
	g->fd = fd;
	g->map = map;
	g->map_size = file_size;

	Reader r = { .base = (const uint8_t *)map, .pos = 0, .size = file_size };

	uint32_t magic = rd_u32(&r);
	if(magic != GGUF_MAGIC)
	{
		fprintf(stderr, "Not a GGUF file\n");
		exit(1);
	}
	g->version = rd_u32(&r);
	g->n_tensors = rd_u64(&r);
	g->n_kv = rd_u64(&r);

	/* Remember KV section start */
	g->kv_section = r.base + r.pos;

	/* Skip all KV pairs */
	for(uint64_t i = 0; i < g->n_kv; i++)
	{
		skip_value(&r, GGUF_TYPE_STRING); /* key */
		uint32_t vtype = rd_u32(&r);
		skip_value(&r, vtype);
	}
	g->kv_section_size = (r.base + r.pos) - g->kv_section;

	/* Parse tensor info */
	g->tensors = calloc(g->n_tensors, sizeof(TensorInfo));
	for(uint64_t i = 0; i < g->n_tensors; i++)
	{
		TensorInfo *t = &g->tensors[i];
		uint64_t name_len = rd_u64(&r);
		if(name_len >= sizeof(t->name))
			name_len = sizeof(t->name) - 1;
		memcpy(t->name, r.base + r.pos, name_len);
		t->name[name_len] = 0;
		r.pos += name_len + (rd_u64(&r), 0); /* name was already read via name_len */
		/* Re-do: name_len was read, now skip name bytes */
	}

	/* Redo tensor parsing properly */
	r.pos = g->kv_section + g->kv_section_size - r.base;
	for(uint64_t i = 0; i < g->n_tensors; i++)
	{
		TensorInfo *t = &g->tensors[i];
		uint64_t name_len = rd_u64(&r);
		size_t copy_len = name_len < 127 ? name_len : 127;
		memcpy(t->name, r.base + r.pos, copy_len);
		t->name[copy_len] = 0;
		r.pos += name_len;
		t->n_dims = (int)rd_u32(&r);
		for(int d = 0; d < t->n_dims; d++)
			t->dims[d] = (int64_t)rd_u64(&r);
		t->type = (int)rd_u32(&r);
		t->offset = (size_t)rd_u64(&r);
	}

	/* Tensor data starts after alignment to 32 bytes */
	size_t align = 32;
	size_t data_offset = (r.pos + align - 1) & ~(align - 1);
	g->data_start = (const uint8_t *)map + data_offset;

	return g;
}

static void gguf_close(GGUFFile *g)
{
	munmap(g->map, g->map_size);
	close(g->fd);
	free(g->tensors);
	free(g);
}

/* Find a tensor by name */
static TensorInfo *gguf_find_tensor(GGUFFile *g, const char *name)
{
	for(uint64_t i = 0; i < g->n_tensors; i++)
	{
		if(strcmp(g->tensors[i].name, name) == 0)
			return &g->tensors[i];
	}
	return NULL;
}

/* Get tensor data pointer */
static const void *gguf_tensor_data(GGUFFile *g, TensorInfo *t)
{
	return g->data_start + t->offset;
}

/* Read GGUF metadata value by key — walks the KV section each time */
static bool gguf_find_key(GGUFFile *g, const char *key, Reader *r_out, int *type_out)
{
	Reader r = { .base = (const uint8_t *)g->map,
				 .pos = (size_t)(g->kv_section - (const uint8_t *)g->map),
				 .size = g->map_size };
	for(uint64_t i = 0; i < g->n_kv; i++)
	{
		uint64_t klen = rd_u64(&r);
		const char *k = (const char *)(r.base + r.pos);
		r.pos += klen;
		uint32_t vtype = rd_u32(&r);
		if(klen == strlen(key) && memcmp(k, key, klen) == 0)
		{
			*r_out = r;
			*type_out = (int)vtype;
			return true;
		}
		skip_value(&r, vtype);
	}
	return false;
}

static uint32_t gguf_get_u32(GGUFFile *g, const char *key)
{
	Reader r;
	int t;
	if(!gguf_find_key(g, key, &r, &t))
	{
		fprintf(stderr, "Key not found: %s\n", key);
		exit(1);
	}
	return rd_u32(&r);
}

static float gguf_get_f32(GGUFFile *g, const char *key)
{
	Reader r;
	int t;
	if(!gguf_find_key(g, key, &r, &t))
	{
		fprintf(stderr, "Key not found: %s\n", key);
		exit(1);
	}
	return rd_f32(&r);
}

static uint32_t gguf_get_u32_or(GGUFFile *g, const char *key, uint32_t def)
{
	Reader r;
	int t;
	if(!gguf_find_key(g, key, &r, &t))
		return def;
	return rd_u32(&r);
}

static float gguf_get_f32_or(GGUFFile *g, const char *key, float def)
{
	Reader r;
	int t;
	if(!gguf_find_key(g, key, &r, &t))
		return def;
	return rd_f32(&r);
}

/* Read bool array element */
static bool gguf_get_arr_bool(GGUFFile *g, const char *key, int idx)
{
	Reader r;
	int t;
	if(!gguf_find_key(g, key, &r, &t))
		return false;
	rd_u32(&r); /* elem type */
	uint64_t count = rd_u64(&r);
	if((uint64_t)idx >= count)
		return false;
	return r.base[r.pos + idx] != 0;
}

/* Read u32 array element */
static uint32_t gguf_get_arr_u32(GGUFFile *g, const char *key, int idx)
{
	Reader r;
	int t;
	if(!gguf_find_key(g, key, &r, &t))
		return 0;
	rd_u32(&r);
	rd_u64(&r);
	uint32_t v;
	memcpy(&v, r.base + r.pos + idx * 4, 4);
	return v;
}

/* Read string array element */
static char *gguf_get_arr_str(GGUFFile *g, const char *key, int idx)
{
	Reader r;
	int t;
	if(!gguf_find_key(g, key, &r, &t))
		return NULL;
	rd_u32(&r);
	uint64_t count = rd_u64(&r);
	for(int i = 0; i < idx && (uint64_t)i < count; i++)
		skip_value(&r, GGUF_TYPE_STRING);
	if((uint64_t)idx >= count)
		return NULL;
	return rd_str_dup(&r);
}

static uint64_t gguf_get_arr_count(GGUFFile *g, const char *key)
{
	Reader r;
	int t;
	if(!gguf_find_key(g, key, &r, &t))
		return 0;
	rd_u32(&r);
	return rd_u64(&r);
}

/* ── Tokenizer (BPE) ────────────────────────────────────────────────── */

typedef struct
{
	int n_vocab;
	char **tokens;
	float *scores;
	int *types;
	int *ht; /* hash table: token_id or -1 */
	int ht_size;
	int bos_id;
	int eos_id;
	/* Byte fallback token IDs */
	int byte_tokens[256];
} Tokenizer;

static uint32_t hash_str(const char *s)
{
	uint32_t h = 0x811c9dc5u;
	for(const unsigned char *p = (const unsigned char *)s; *p; p++)
		h = (h ^ *p) * 0x01000193u;
	return h;
}

static int tok_lookup(Tokenizer *tok, const char *s)
{
	uint32_t h = hash_str(s) % (uint32_t)tok->ht_size;
	for(;;)
	{
		int id = tok->ht[h];
		if(id < 0)
			return -1;
		if(strcmp(tok->tokens[id], s) == 0)
			return id;
		h = (h + 1) % (uint32_t)tok->ht_size;
	}
}

static void tok_insert(Tokenizer *tok, int id)
{
	uint32_t h = hash_str(tok->tokens[id]) % (uint32_t)tok->ht_size;
	while(tok->ht[h] >= 0)
		h = (h + 1) % (uint32_t)tok->ht_size;
	tok->ht[h] = id;
}

static Tokenizer *tokenizer_init(GGUFFile *g)
{
	Tokenizer *tok = calloc(1, sizeof(Tokenizer));
	tok->n_vocab = (int)gguf_get_arr_count(g, "tokenizer.ggml.tokens");
	tok->bos_id = (int)gguf_get_u32(g, "tokenizer.ggml.bos_token_id");
	tok->eos_id = (int)gguf_get_u32(g, "tokenizer.ggml.eos_token_id");

	/* Allocate */
	tok->tokens = calloc(tok->n_vocab, sizeof(char *));
	tok->scores = calloc(tok->n_vocab, sizeof(float));
	tok->types = calloc(tok->n_vocab, sizeof(int));

	/* Hash table: 3x vocab for low load factor */
	tok->ht_size = tok->n_vocab * 3;
	tok->ht = malloc(tok->ht_size * sizeof(int));
	for(int i = 0; i < tok->ht_size; i++)
		tok->ht[i] = -1;

	/* Read tokens */
	printf("Loading tokenizer (%d tokens)...\n", tok->n_vocab);

	/* Read all tokens from GGUF array */
	Reader r;
	int t;
	gguf_find_key(g, "tokenizer.ggml.tokens", &r, &t);
	rd_u32(&r); /* elem type */
	uint64_t count = rd_u64(&r);
	for(uint64_t i = 0; i < count && (int)i < tok->n_vocab; i++)
	{
		tok->tokens[i] = rd_str_dup(&r);
	}

	/* Read scores */
	gguf_find_key(g, "tokenizer.ggml.scores", &r, &t);
	rd_u32(&r);
	count = rd_u64(&r);
	for(uint64_t i = 0; i < count && (int)i < tok->n_vocab; i++)
	{
		tok->scores[i] = rd_f32(&r);
	}

	/* Read types */
	gguf_find_key(g, "tokenizer.ggml.token_type", &r, &t);
	rd_u32(&r);
	count = rd_u64(&r);
	for(uint64_t i = 0; i < count && (int)i < tok->n_vocab; i++)
	{
		tok->types[i] = rd_i32(&r);
	}

	/* Build hash table */
	for(int i = 0; i < tok->n_vocab; i++)
	{
		if(tok->tokens[i] && tok->tokens[i][0])
			tok_insert(tok, i);
	}

	/* Find byte fallback tokens (<0x00> through <0xFF>) */
	memset(tok->byte_tokens, -1, sizeof(tok->byte_tokens));
	char buf[8];
	for(int b = 0; b < 256; b++)
	{
		snprintf(buf, sizeof(buf), "<0x%02X>", b);
		int id = tok_lookup(tok, buf);
		if(id >= 0)
			tok->byte_tokens[b] = id;
	}

	return tok;
}

/* UTF-8: number of bytes for character starting with byte c */
static int utf8_len(unsigned char c)
{
	if(c < 0x80)
		return 1;
	if(c < 0xE0)
		return 2;
	if(c < 0xF0)
		return 3;
	return 4;
}

/* Encode text to token IDs using BPE */
static int tokenize(Tokenizer *tok, const char *text, int *tokens_out, int max_tokens)
{
	/* Step 1: Replace spaces with ▁ (U+2581 = 0xE2 0x96 0x81) */
	size_t tlen = strlen(text);
	char *buf = malloc(tlen * 3 + 1);
	size_t bpos = 0;
	for(size_t i = 0; i < tlen; i++)
	{
		if(text[i] == ' ')
		{
			buf[bpos++] = (char)0xE2;
			buf[bpos++] = (char)0x96;
			buf[bpos++] = (char)0x81;
		}
		else
		{
			buf[bpos++] = text[i];
		}
	}
	buf[bpos] = 0;

	/* Step 2: Split into initial character tokens */
	typedef struct
	{
		char *str;
		int id;
	} BPEToken;
	int n_tok = 0;
	int cap = (int)bpos + 16;
	BPEToken *toks = malloc(cap * sizeof(BPEToken));

	size_t p = 0;
	while(p < bpos)
	{
		int clen = utf8_len((unsigned char)buf[p]);
		if(p + clen > bpos)
			clen = (int)(bpos - p);
		char tmp[8];
		memcpy(tmp, buf + p, clen);
		tmp[clen] = 0;

		int id = tok_lookup(tok, tmp);
		if(id >= 0)
		{
			toks[n_tok].str = strdup(tmp);
			toks[n_tok].id = id;
		}
		else
		{
			/* Byte fallback */
			for(int b = 0; b < clen; b++)
			{
				if(n_tok >= cap)
				{
					cap *= 2;
					toks = realloc(toks, cap * sizeof(BPEToken));
				}
				unsigned char byte = (unsigned char)buf[p + b];
				char fb[8];
				snprintf(fb, sizeof(fb), "<0x%02X>", byte);
				toks[n_tok].str = strdup(fb);
				toks[n_tok].id = tok->byte_tokens[byte];
				n_tok++;
			}
			p += clen;
			continue;
		}
		n_tok++;
		p += clen;
	}
	free(buf);

	/* Step 3: BPE merge loop — find best merge (highest score) and apply */
	for(;;)
	{
		float best_score = -1e30f;
		int best_idx = -1;
		int best_id = -1;

		for(int i = 0; i < n_tok - 1; i++)
		{
			/* Try merging toks[i] and toks[i+1] */
			size_t l1 = strlen(toks[i].str);
			size_t l2 = strlen(toks[i + 1].str);
			char *merged = malloc(l1 + l2 + 1);
			memcpy(merged, toks[i].str, l1);
			memcpy(merged + l1, toks[i + 1].str, l2);
			merged[l1 + l2] = 0;

			int id = tok_lookup(tok, merged);
			if(id >= 0 && tok->scores[id] > best_score)
			{
				best_score = tok->scores[id];
				best_idx = i;
				best_id = id;
			}
			free(merged);
		}

		if(best_idx < 0)
			break;

		/* Apply merge */
		size_t l1 = strlen(toks[best_idx].str);
		size_t l2 = strlen(toks[best_idx + 1].str);
		char *merged = malloc(l1 + l2 + 1);
		memcpy(merged, toks[best_idx].str, l1);
		memcpy(merged + l1, toks[best_idx + 1].str, l2);
		merged[l1 + l2] = 0;

		free(toks[best_idx].str);
		free(toks[best_idx + 1].str);
		toks[best_idx].str = merged;
		toks[best_idx].id = best_id;

		/* Shift remaining tokens left */
		for(int i = best_idx + 1; i < n_tok - 1; i++)
			toks[i] = toks[i + 1];
		n_tok--;
	}

	/* Output token IDs (no BOS here — caller adds it) */
	int out_len = 0;
	for(int i = 0; i < n_tok && out_len < max_tokens; i++)
	{
		tokens_out[out_len++] = toks[i].id;
		free(toks[i].str);
	}
	free(toks);
	return out_len;
}

/* Special token IDs for Gemma 4 chat */
#define TOK_THINK		98	/* <|think|> — enables thinking mode */
#define TOK_CHANNEL		100 /* <|channel> — start channel block */
#define TOK_CHANNEL_END 101 /* <channel|> — end channel block */
#define TOK_START_TURN	105 /* <|turn> */
#define TOK_END_TURN	106 /* <turn|> */
#define TOK_NEWLINE		107 /* \n */

/* Tokenize with chat template for instruction-tuned models.
 *
 * With thinking (default):
 *   <bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n{msg}<turn|>\n<|turn>model\n
 *
 * Without thinking:
 *   <bos><|turn>user\n{msg}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
 */
static int tokenize_chat(Tokenizer *tok, const char *text, int *tokens_out, int max_tokens, bool think)
{
	int pos = 0;
	tokens_out[pos++] = tok->bos_id; /* <bos> */

	if(think)
	{
		/* System turn with thinking flag */
		tokens_out[pos++] = TOK_START_TURN;
		pos += tokenize(tok, "system", tokens_out + pos, max_tokens - pos);
		tokens_out[pos++] = TOK_NEWLINE;
		tokens_out[pos++] = TOK_THINK; /* <|think|> */
		tokens_out[pos++] = TOK_NEWLINE;
		tokens_out[pos++] = TOK_END_TURN;
		tokens_out[pos++] = TOK_NEWLINE;
	}

	/* User turn */
	tokens_out[pos++] = TOK_START_TURN;
	pos += tokenize(tok, "user", tokens_out + pos, max_tokens - pos);
	tokens_out[pos++] = TOK_NEWLINE;
	pos += tokenize(tok, text, tokens_out + pos, max_tokens - pos);
	tokens_out[pos++] = TOK_END_TURN;
	tokens_out[pos++] = TOK_NEWLINE;

	/* Model turn */
	tokens_out[pos++] = TOK_START_TURN;
	pos += tokenize(tok, "model", tokens_out + pos, max_tokens - pos);
	tokens_out[pos++] = TOK_NEWLINE;

	if(!think)
	{
		/* Suppress thinking with empty thought channel */
		tokens_out[pos++] = TOK_CHANNEL;
		pos += tokenize(tok, "thought", tokens_out + pos, max_tokens - pos);
		tokens_out[pos++] = TOK_NEWLINE;
		tokens_out[pos++] = TOK_CHANNEL_END;
	}

	return pos;
}

/* Tokenize raw text (with BOS) */
static int tokenize_raw(Tokenizer *tok, const char *text, int *tokens_out, int max_tokens)
{
	int pos = 0;
	tokens_out[pos++] = tok->bos_id;
	pos += tokenize(tok, text, tokens_out + pos, max_tokens - pos);
	return pos;
}

static const char *detokenize(Tokenizer *tok, int id)
{
	if(id < 0 || id >= tok->n_vocab)
		return "";
	return tok->tokens[id];
}

/* Print a token, converting ▁ back to space and handling special tokens */
static bool in_thinking = false;

static void print_token(Tokenizer *tok, int id)
{
	/* Handle thinking channel markers */
	if(id == TOK_CHANNEL)
	{
		in_thinking = true;
		printf("[thinking] ");
		return;
	}
	if(id == TOK_CHANNEL_END)
	{
		if(in_thinking)
		{
			printf(" [/thinking]\n");
			in_thinking = false;
		}
		return;
	}

	const char *s = detokenize(tok, id);
	if(!s)
		return;
	/* Handle byte tokens <0xHH> */
	if(s[0] == '<' && s[1] == '0' && s[2] == 'x' && strlen(s) == 6 && s[5] == '>')
	{
		unsigned int byte;
		sscanf(s + 3, "%02X", &byte);
		putchar((char)byte);
		return;
	}
	/* Replace ▁ with space */
	while(*s)
	{
		if((unsigned char)s[0] == 0xE2 && (unsigned char)s[1] == 0x96 && (unsigned char)s[2] == 0x81)
		{
			putchar(' ');
			s += 3;
		}
		else
		{
			putchar(*s);
			s++;
		}
	}
}

/* ── Model ───────────────────────────────────────────────────────────── */

typedef struct
{
	const void *data;
	int type;
	int64_t dim0; /* inner (contiguous) dimension */
	int64_t dim1; /* outer dimension (rows) */
} Tensor;

typedef struct
{
	/* Attention */
	Tensor attn_norm;
	Tensor attn_q, attn_k, attn_v, attn_output;
	Tensor attn_q_norm, attn_k_norm;
	Tensor post_attn_norm;
	/* FFN */
	Tensor ffn_norm;
	Tensor ffn_gate, ffn_up, ffn_down;
	Tensor post_ffn_norm;
	/* PLE */
	Tensor ple_inp_gate, ple_proj, ple_post_norm;
	float layer_scale;
	bool has_kv;
	bool is_swa;
	int head_dim;
	int n_kv_head;
	int n_ff;
} Layer;

typedef struct
{
	/* Hyperparams */
	int n_embd, n_head, n_kv_head_swa, n_kv_head_full;
	int n_layer, n_vocab;
	int head_dim_swa, head_dim_full;
	int sliding_window;
	int n_kv_shared;
	int n_embd_per_layer;
	float rms_eps, softcap;
	float rope_theta_full, rope_theta_swa;
	int max_ctx;

	/* Layer config */
	bool swa_pattern[MAX_LAYERS];
	int n_ff[MAX_LAYERS];
	int kv_source[MAX_LAYERS]; /* which layer's KV cache to read from */

	/* Global tensors */
	Tensor token_embd, output_norm;
	Tensor ple_token_embd, ple_model_proj, ple_proj_norm;
	const float *rope_freqs; /* [head_dim_full / 2] frequency factors */

	/* Per-layer */
	Layer layers[MAX_LAYERS];

	/* KV caches: only for layers that have own KV */
	float *kv_k[MAX_LAYERS]; /* [max_kv_len, head_dim] */
	float *kv_v[MAX_LAYERS];
	int kv_max_len[MAX_LAYERS];

	/* Scratch buffers */
	float *x, *xb, *xb2;
	float *q_buf, *k_buf, *v_buf;
	float *att;
	float *hb, *hb2;
	float *logits;
	float *ple_buf; /* [n_embd_per_layer * n_layer] */
	float *ple_tmp; /* [n_embd_per_layer] */
	float *dq_buf;	/* dequantization scratch: max row size */

	/* Tokenizer */
	Tokenizer *tok;
} Model;

static Tensor load_tensor(GGUFFile *g, const char *name, bool required)
{
	TensorInfo *ti = gguf_find_tensor(g, name);
	if(!ti)
	{
		if(required)
		{
			fprintf(stderr, "Required tensor not found: %s\n", name);
			exit(1);
		}
		return (Tensor) { .data = NULL, .type = -1, .dim0 = 0, .dim1 = 0 };
	}
	return (Tensor) { .data = gguf_tensor_data(g, ti),
					  .type = ti->type,
					  .dim0 = ti->dims[0],
					  .dim1 = ti->n_dims > 1 ? ti->dims[1] : 1 };
}

static Model *model_load(const char *path)
{
	printf("Loading model from %s...\n", path);
	GGUFFile *g = gguf_open(path);

	Model *m = calloc(1, sizeof(Model));

	/* Read hyperparameters */
	m->n_layer = (int)gguf_get_u32(g, "gemma4.block_count");
	m->n_embd = (int)gguf_get_u32(g, "gemma4.embedding_length");
	m->n_head = (int)gguf_get_u32(g, "gemma4.attention.head_count");
	m->n_kv_head_swa = (int)gguf_get_u32(g, "gemma4.attention.head_count_kv");
	m->n_vocab = (int)gguf_get_arr_count(g, "tokenizer.ggml.tokens");
	m->head_dim_swa = (int)gguf_get_u32_or(g, "gemma4.attention.key_length_swa", 256);
	m->head_dim_full = (int)gguf_get_u32_or(g, "gemma4.attention.key_length", 512);
	m->sliding_window = (int)gguf_get_u32_or(g, "gemma4.attention.sliding_window", 512);
	m->n_kv_shared = (int)gguf_get_u32_or(g, "gemma4.attention.shared_kv_layers", 0);
	m->n_embd_per_layer = (int)gguf_get_u32_or(g, "gemma4.embedding_length_per_layer_input", 0);
	m->rms_eps = gguf_get_f32_or(g, "gemma4.attention.layer_norm_rms_epsilon", 1e-6f);
	m->softcap = gguf_get_f32_or(g, "gemma4.final_logit_softcapping", 30.0f);
	m->rope_theta_full = gguf_get_f32_or(g, "gemma4.rope.freq_base", 1000000.0f);
	m->rope_theta_swa = gguf_get_f32_or(g, "gemma4.rope.freq_base_swa", 10000.0f);
	m->n_kv_head_full = m->n_kv_head_swa; /* For E2B, same; override if metadata exists */

	int n_layer_kv = m->n_layer - m->n_kv_shared;

	printf("  n_layer=%d, n_embd=%d, n_head=%d, n_kv_head=%d\n", m->n_layer, m->n_embd, m->n_head, m->n_kv_head_swa);
	printf("  head_dim_swa=%d, head_dim_full=%d, sliding_window=%d\n",
		   m->head_dim_swa,
		   m->head_dim_full,
		   m->sliding_window);
	printf("  n_kv_shared=%d, n_embd_per_layer=%d\n", m->n_kv_shared, m->n_embd_per_layer);

	/* Read per-layer SWA pattern */
	for(int il = 0; il < m->n_layer; il++)
	{
		m->swa_pattern[il] = gguf_get_arr_bool(g, "gemma4.attention.sliding_window_pattern", il);
	}

	/* Read per-layer FFN sizes */
	for(int il = 0; il < m->n_layer; il++)
	{
		m->n_ff[il] = (int)gguf_get_arr_u32(g, "gemma4.feed_forward_length", il);
	}

	/* Build KV source mapping */
	for(int il = 0; il < m->n_layer; il++)
	{
		if(il < n_layer_kv)
		{
			m->kv_source[il] = il; /* own KV */
		}
		else if(m->swa_pattern[il])
		{
			m->kv_source[il] = n_layer_kv - 2; /* last SWA layer with own KV */
		}
		else
		{
			m->kv_source[il] = n_layer_kv - 1; /* last full layer with own KV */
		}
	}

	/* Load global tensors */
	m->token_embd = load_tensor(g, "token_embd.weight", true);
	m->output_norm = load_tensor(g, "output_norm.weight", true);

	/* RoPE frequency factors */
	TensorInfo *rf = gguf_find_tensor(g, "rope_freqs.weight");
	if(rf)
		m->rope_freqs = (const float *)gguf_tensor_data(g, rf);

	/* PLE global tensors */
	if(m->n_embd_per_layer > 0)
	{
		m->ple_token_embd = load_tensor(g, "per_layer_token_embd.weight", true);
		m->ple_model_proj = load_tensor(g, "per_layer_model_proj.weight", true);
		m->ple_proj_norm = load_tensor(g, "per_layer_proj_norm.weight", true);
	}

	/* Load per-layer tensors */
	for(int il = 0; il < m->n_layer; il++)
	{
		Layer *l = &m->layers[il];
		char name[128];

		l->is_swa = m->swa_pattern[il];
		l->has_kv = (il < n_layer_kv);
		l->head_dim = l->is_swa ? m->head_dim_swa : m->head_dim_full;
		l->n_kv_head = l->is_swa ? m->n_kv_head_swa : m->n_kv_head_full;
		l->n_ff = m->n_ff[il];

#define LOAD(field, suffix, req)                            \
	do                                                      \
	{                                                       \
		snprintf(name, sizeof(name), "blk.%d." suffix, il); \
		l->field = load_tensor(g, name, req);               \
	} while(0)

		LOAD(attn_norm, "attn_norm.weight", true);
		LOAD(attn_q, "attn_q.weight", true);
		LOAD(attn_k, "attn_k.weight", false);
		LOAD(attn_v, "attn_v.weight", false);
		LOAD(attn_q_norm, "attn_q_norm.weight", true);
		LOAD(attn_k_norm, "attn_k_norm.weight", false);
		LOAD(attn_output, "attn_output.weight", true);
		LOAD(post_attn_norm, "post_attention_norm.weight", true);
		LOAD(ffn_norm, "ffn_norm.weight", true);
		LOAD(ffn_gate, "ffn_gate.weight", true);
		LOAD(ffn_up, "ffn_up.weight", true);
		LOAD(ffn_down, "ffn_down.weight", true);
		LOAD(post_ffn_norm, "post_ffw_norm.weight", true);

		if(m->n_embd_per_layer > 0)
		{
			LOAD(ple_inp_gate, "inp_gate.weight", true);
			LOAD(ple_proj, "proj.weight", true);
			LOAD(ple_post_norm, "post_norm.weight", true);
		}
#undef LOAD

		/* Layer output scale */
		snprintf(name, sizeof(name), "blk.%d.layer_output_scale.weight", il);
		TensorInfo *st = gguf_find_tensor(g, name);
		if(st)
		{
			const float *sd = (const float *)gguf_tensor_data(g, st);
			l->layer_scale = sd[0];
		}
		else
		{
			l->layer_scale = 1.0f;
		}
	}

	/* Initialize tokenizer */
	m->tok = tokenizer_init(g);

	/* Note: we keep the GGUF mmap alive for the lifetime of the model */
	printf("Model loaded successfully.\n");
	return m;
}

/* Allocate all scratch buffers and KV caches */
static void model_init_state(Model *m, int max_ctx)
{
	m->max_ctx = max_ctx;
	int ne = m->n_embd;
	int max_hd = m->head_dim_full > m->head_dim_swa ? m->head_dim_full : m->head_dim_swa;
	int max_ff = 0;
	for(int il = 0; il < m->n_layer; il++)
		if(m->n_ff[il] > max_ff)
			max_ff = m->n_ff[il];

	m->x = calloc(ne, sizeof(float));
	m->xb = calloc(ne, sizeof(float));
	m->xb2 = calloc(ne, sizeof(float));
	m->q_buf = calloc(m->n_head * max_hd, sizeof(float));
	m->k_buf = calloc(m->n_kv_head_swa * max_hd, sizeof(float)); /* max 1 KV head for E2B */
	m->v_buf = calloc(m->n_kv_head_swa * max_hd, sizeof(float));
	m->att = calloc(m->n_head * (size_t)max_ctx, sizeof(float));
	m->hb = calloc(max_ff, sizeof(float));
	m->hb2 = calloc(max_ff, sizeof(float));
	m->logits = calloc(m->n_vocab, sizeof(float));

	/* Dequantization scratch: enough for largest row */
	int max_row = m->n_vocab; /* embedding rows are largest */
	if(max_ff > max_row)
		max_row = max_ff;
	m->dq_buf = calloc(max_row, sizeof(float));

	/* PLE buffers */
	if(m->n_embd_per_layer > 0)
	{
		m->ple_buf = calloc(m->n_embd_per_layer * m->n_layer, sizeof(float));
		m->ple_tmp = calloc(m->n_embd_per_layer, sizeof(float));
	}

	/* Allocate KV caches */
	int n_layer_kv = m->n_layer - m->n_kv_shared;
	for(int il = 0; il < n_layer_kv; il++)
	{
		Layer *l = &m->layers[il];
		int kv_len = l->is_swa ? m->sliding_window : max_ctx;
		int kv_dim = l->n_kv_head * l->head_dim;
		m->kv_k[il] = calloc((size_t)kv_len * kv_dim, sizeof(float));
		m->kv_v[il] = calloc((size_t)kv_len * kv_dim, sizeof(float));
		m->kv_max_len[il] = kv_len;
	}

	printf("State initialized (max_ctx=%d, KV caches for %d layers).\n", max_ctx, n_layer_kv);
}

/* ── Math operations ─────────────────────────────────────────────────── */

static void rmsnorm(float *out, const float *x, const float *weight, int n, float eps)
{
	float ss = 0.0f;
	for(int i = 0; i < n; i++)
		ss += x[i] * x[i];
	ss = 1.0f / sqrtf(ss / (float)n + eps);
	for(int i = 0; i < n; i++)
		out[i] = x[i] * ss * weight[i];
}

static void rmsnorm_f32weight(float *out, const float *x, const void *weight_data, int n, float eps)
{
	const float *w = (const float *)weight_data;
	rmsnorm(out, x, w, n, eps);
}

static void rmsnorm_noweight(float *out, const float *x, int n, float eps)
{
	float ss = 0.0f;
	for(int i = 0; i < n; i++)
		ss += x[i] * x[i];
	ss = 1.0f / sqrtf(ss / (float)n + eps);
	for(int i = 0; i < n; i++)
		out[i] = x[i] * ss;
}

static void softmax(float *x, int n)
{
	float max_val = x[0];
	for(int i = 1; i < n; i++)
		if(x[i] > max_val)
			max_val = x[i];
	float sum = 0.0f;
	for(int i = 0; i < n; i++)
	{
		x[i] = expf(x[i] - max_val);
		sum += x[i];
	}
	for(int i = 0; i < n; i++)
		x[i] /= sum;
}

static float gelu_tanh(float x)
{
	return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

/* Matrix-vector multiply: out[i] = sum_j(W[i,j] * x[j]) */
/* W is stored as dim1 rows of dim0 elements each, quantized */
static void matmul(float *out, Tensor w, const float *x, float *dq_buf)
{
	int n_out = (int)w.dim1;
	int n_in = (int)w.dim0;
	size_t rb = row_bytes(n_in, w.type);

	for(int i = 0; i < n_out; i++)
	{
		const void *row_ptr = (const uint8_t *)w.data + (size_t)i * rb;
		dequantize_row(row_ptr, dq_buf, n_in, w.type);
		float sum = 0.0f;
		for(int j = 0; j < n_in; j++)
			sum += dq_buf[j] * x[j];
		out[i] = sum;
	}
}

/* Apply RoPE to a vector (NEOX style) */
static void apply_rope(float *vec, int head_dim, int n_rot, int pos, float theta_base, const float *freq_factors)
{
	int half = head_dim / 2;
	int n_pairs = n_rot / 2;
	for(int i = 0; i < n_pairs; i++)
	{
		float freq = powf(theta_base, -2.0f * (float)i / (float)n_rot);
		if(freq_factors)
			freq *= freq_factors[i];
		float angle = (float)pos * freq;
		float cos_a = cosf(angle);
		float sin_a = sinf(angle);
		float x0 = vec[i];
		float x1 = vec[i + half];
		vec[i] = x0 * cos_a - x1 * sin_a;
		vec[i + half] = x0 * sin_a + x1 * cos_a;
	}
}

/* ── Forward pass ────────────────────────────────────────────────────── */

static float *forward(Model *m, int token, int pos)
{
	int ne = m->n_embd;
	int nh = m->n_head;
	float eps = m->rms_eps;

	/* Token embedding lookup: dequantize row `token` from token_embd */
	{
		size_t rb = row_bytes(ne, m->token_embd.type);
		const void *row = (const uint8_t *)m->token_embd.data + (size_t)token * rb;
		dequantize_row(row, m->x, ne, m->token_embd.type);
	}

	/* Scale embedding by sqrt(n_embd) */
	float embd_scale = sqrtf((float)ne);
	for(int i = 0; i < ne; i++)
		m->x[i] *= embd_scale;

	/* Compute per-layer embeddings (PLE) */
	if(m->n_embd_per_layer > 0)
	{
		int npl = m->n_embd_per_layer;
		int total_ple = npl * m->n_layer;

		/* 1. Token embedding path: lookup row `token` from ple_token_embd → [total_ple] */
		float *ple_tok = m->ple_buf;
		{
			size_t rb = row_bytes(total_ple, m->ple_token_embd.type);
			const void *row = (const uint8_t *)m->ple_token_embd.data + (size_t)token * rb;
			dequantize_row(row, ple_tok, total_ple, m->ple_token_embd.type);
		}
		float tok_scale = sqrtf((float)npl);
		for(int i = 0; i < total_ple; i++)
			ple_tok[i] *= tok_scale;

		/* 2. Model projection path: ple_model_proj @ x → [total_ple] */
		float *ple_proj = calloc(total_ple, sizeof(float));
		matmul(ple_proj, m->ple_model_proj, m->x, m->dq_buf);
		float proj_scale = 1.0f / sqrtf((float)ne);
		for(int i = 0; i < total_ple; i++)
			ple_proj[i] *= proj_scale;

		/* RMSNorm the projection (per-layer chunks) */
		const float *pn = (const float *)m->ple_proj_norm.data;
		for(int il = 0; il < m->n_layer; il++)
		{
			float *chunk = ple_proj + il * npl;
			float ss = 0.0f;
			for(int j = 0; j < npl; j++)
				ss += chunk[j] * chunk[j];
			ss = 1.0f / sqrtf(ss / (float)npl + eps);
			for(int j = 0; j < npl; j++)
				chunk[j] = chunk[j] * ss * pn[j];
		}

		/* 3. Combine: add and scale */
		float comb_scale = 1.0f / sqrtf(2.0f);
		for(int i = 0; i < total_ple; i++)
			m->ple_buf[i] = (ple_tok[i] + ple_proj[i]) * comb_scale;

		free(ple_proj);
	}

	/* ── Layer loop ── */
	for(int il = 0; il < m->n_layer; il++)
	{
		Layer *l = &m->layers[il];
		int hd = l->head_dim;
		int n_kv = l->n_kv_head;
		int kv_src = m->kv_source[il];

		/* ── Attention ── */

		/* Pre-attention RMSNorm */
		rmsnorm_f32weight(m->xb, m->x, l->attn_norm.data, ne, eps);

		/* Q projection: [ne] → [nh * hd] */
		matmul(m->q_buf, l->attn_q, m->xb, m->dq_buf);

		/* Per-head Q normalization + RoPE */
		const float *qnw = (const float *)l->attn_q_norm.data;
		float theta = l->is_swa ? m->rope_theta_swa : m->rope_theta_full;
		int n_rot = l->is_swa ? m->head_dim_swa : (m->head_dim_full / 4); /* partial_rotary_factor=0.25 for full */
		const float *ff = l->is_swa ? NULL : m->rope_freqs;

		for(int h = 0; h < nh; h++)
		{
			float *qh = m->q_buf + h * hd;
			/* RMSNorm per head */
			float ss = 0.0f;
			for(int j = 0; j < hd; j++)
				ss += qh[j] * qh[j];
			ss = 1.0f / sqrtf(ss / (float)hd + eps);
			for(int j = 0; j < hd; j++)
				qh[j] = qh[j] * ss * qnw[j];
			/* RoPE */
			apply_rope(qh, hd, n_rot, pos, theta, ff);
		}

		/* K, V projection + normalization + RoPE + cache write */
		if(l->has_kv && l->attn_k.data && l->attn_v.data)
		{
			matmul(m->k_buf, l->attn_k, m->xb, m->dq_buf);
			matmul(m->v_buf, l->attn_v, m->xb, m->dq_buf);

			/* K normalization + RoPE per KV head */
			const float *knw = (const float *)l->attn_k_norm.data;
			for(int h = 0; h < n_kv; h++)
			{
				float *kh = m->k_buf + h * hd;
				float ss = 0.0f;
				for(int j = 0; j < hd; j++)
					ss += kh[j] * kh[j];
				ss = 1.0f / sqrtf(ss / (float)hd + eps);
				for(int j = 0; j < hd; j++)
					kh[j] = kh[j] * ss * knw[j];
				apply_rope(kh, hd, n_rot, pos, theta, ff);
			}

			/* V normalization (no learned weight) */
			for(int h = 0; h < n_kv; h++)
			{
				rmsnorm_noweight(m->v_buf + h * hd, m->v_buf + h * hd, hd, eps);
			}

			/* Write to KV cache */
			int cache_idx;
			if(l->is_swa)
			{
				cache_idx = pos % m->sliding_window;
			}
			else
			{
				cache_idx = pos;
				if(cache_idx >= m->kv_max_len[il])
				{
					fprintf(stderr, "Context overflow at layer %d, pos %d\n", il, pos);
					exit(1);
				}
			}
			int kv_dim = n_kv * hd;
			memcpy(m->kv_k[il] + cache_idx * kv_dim, m->k_buf, kv_dim * sizeof(float));
			memcpy(m->kv_v[il] + cache_idx * kv_dim, m->v_buf, kv_dim * sizeof(float));
		}

		/* Attention computation: Q @ K^T, softmax, @ V */
		{
			int src_hd = m->layers[kv_src].head_dim;
			int src_n_kv = m->layers[kv_src].n_kv_head;
			bool src_swa = m->layers[kv_src].is_swa;
			int kv_dim = src_n_kv * src_hd;

			int kv_len, kv_start;
			if(l->is_swa)
			{
				kv_start = pos >= m->sliding_window ? pos - m->sliding_window + 1 : 0;
				kv_len = pos - kv_start + 1;
			}
			else
			{
				kv_start = 0;
				kv_len = pos + 1;
			}

			/* For each query head */
			for(int h = 0; h < nh; h++)
			{
				float *qh = m->q_buf + h * hd;
				int kv_h = h / (nh / src_n_kv); /* GQA mapping */
				float *scores = m->att + h * m->max_ctx;

				/* Compute attention scores */
				for(int t = 0; t < kv_len; t++)
				{
					int cache_pos = kv_start + t;
					int cache_idx = src_swa ? (cache_pos % m->sliding_window) : cache_pos;
					const float *kh = m->kv_k[kv_src] + cache_idx * kv_dim + kv_h * src_hd;

					float score = 0.0f;
					/* Note: Q has hd dims, K has src_hd dims. They should match for same attention type. */
					int dot_dim = hd < src_hd ? hd : src_hd;
					for(int j = 0; j < dot_dim; j++)
						score += qh[j] * kh[j];
					scores[t] = score; /* attention_scale = 1.0 (no 1/sqrt(d_k)) */
				}

				softmax(scores, kv_len);

				/* Weighted sum of V → output for this head */
				float *out_h = m->xb2 + h * hd; /* reuse xb2 temporarily */
				memset(out_h, 0, hd * sizeof(float));
				for(int t = 0; t < kv_len; t++)
				{
					int cache_pos = kv_start + t;
					int cache_idx = src_swa ? (cache_pos % m->sliding_window) : cache_pos;
					const float *vh = m->kv_v[kv_src] + cache_idx * kv_dim + kv_h * src_hd;
					float w = scores[t];
					int val_dim = hd < src_hd ? hd : src_hd;
					for(int j = 0; j < val_dim; j++)
						out_h[j] += w * vh[j];
				}
			}
		}

		/* Output projection: attn_output @ concat(heads) → [ne] */
		/* xb2 currently holds [nh * hd] concatenated head outputs */
		matmul(m->xb, l->attn_output, m->xb2, m->dq_buf);

		/* Post-attention norm */
		rmsnorm_f32weight(m->xb, m->xb, l->post_attn_norm.data, ne, eps);

		/* Residual add */
		for(int i = 0; i < ne; i++)
			m->x[i] += m->xb[i];

		/* ── Feed-Forward Network (GeGLU) ── */

		/* Pre-FFN RMSNorm */
		rmsnorm_f32weight(m->xb, m->x, l->ffn_norm.data, ne, eps);

		/* gate_proj and up_proj */
		matmul(m->hb, l->ffn_gate, m->xb, m->dq_buf);
		matmul(m->hb2, l->ffn_up, m->xb, m->dq_buf);

		/* GeGLU: gelu(gate) * up */
		for(int i = 0; i < l->n_ff; i++)
			m->hb[i] = gelu_tanh(m->hb[i]) * m->hb2[i];

		/* down_proj */
		matmul(m->xb, l->ffn_down, m->hb, m->dq_buf);

		/* Post-FFN norm */
		rmsnorm_f32weight(m->xb, m->xb, l->post_ffn_norm.data, ne, eps);

		/* Residual add */
		for(int i = 0; i < ne; i++)
			m->x[i] += m->xb[i];

		/* ── PLE injection ── */
		if(m->n_embd_per_layer > 0)
		{
			int npl = m->n_embd_per_layer;

			/* Gate: inp_gate @ x → [npl], then GELU */
			matmul(m->ple_tmp, l->ple_inp_gate, m->x, m->dq_buf);
			for(int j = 0; j < npl; j++)
				m->ple_tmp[j] = gelu_tanh(m->ple_tmp[j]);

			/* Element-wise multiply with per-layer embedding */
			float *ple_il = m->ple_buf + il * npl;
			for(int j = 0; j < npl; j++)
				m->ple_tmp[j] *= ple_il[j];

			/* Project back: proj @ ple_tmp → [ne] */
			matmul(m->xb, l->ple_proj, m->ple_tmp, m->dq_buf);

			/* Post-norm */
			rmsnorm_f32weight(m->xb, m->xb, l->ple_post_norm.data, ne, eps);

			/* Residual add */
			for(int i = 0; i < ne; i++)
				m->x[i] += m->xb[i];
		}

		/* Layer output scale */
		if(l->layer_scale != 1.0f)
		{
			for(int i = 0; i < ne; i++)
				m->x[i] *= l->layer_scale;
		}
	}

	/* Final RMSNorm */
	rmsnorm_f32weight(m->x, m->x, m->output_norm.data, ne, eps);

	/* Output projection (tied weights: uses token_embd) */
	matmul(m->logits, m->token_embd, m->x, m->dq_buf);

	/* Logit soft-capping */
	if(m->softcap > 0.0f)
	{
		for(int i = 0; i < m->n_vocab; i++)
		{
			m->logits[i] = tanhf(m->logits[i] / m->softcap) * m->softcap;
		}
	}

	return m->logits;
}

/* ── Sampling ────────────────────────────────────────────────────────── */

static int sample_argmax(const float *logits, int n)
{
	int best = 0;
	for(int i = 1; i < n; i++)
		if(logits[i] > logits[best])
			best = i;
	return best;
}

typedef struct
{
	float val;
	int idx;
} ProbIndex;

static int cmp_prob_desc(const void *a, const void *b)
{
	float va = ((const ProbIndex *)a)->val;
	float vb = ((const ProbIndex *)b)->val;
	return (va < vb) - (va > vb);
}

static int sample(const float *logits, int n_vocab, float temp, int top_k)
{
	if(temp <= 0.0f)
		return sample_argmax(logits, n_vocab);

	/* Apply temperature */
	static ProbIndex *probs = NULL;
	static int probs_cap = 0;
	if(n_vocab > probs_cap)
	{
		probs_cap = n_vocab;
		probs = realloc(probs, probs_cap * sizeof(ProbIndex));
	}
	for(int i = 0; i < n_vocab; i++)
	{
		probs[i].val = logits[i] / temp;
		probs[i].idx = i;
	}

	/* Softmax */
	float max_val = probs[0].val;
	for(int i = 1; i < n_vocab; i++)
		if(probs[i].val > max_val)
			max_val = probs[i].val;
	float sum = 0.0f;
	for(int i = 0; i < n_vocab; i++)
	{
		probs[i].val = expf(probs[i].val - max_val);
		sum += probs[i].val;
	}
	for(int i = 0; i < n_vocab; i++)
		probs[i].val /= sum;

	/* Top-K */
	if(top_k > 0 && top_k < n_vocab)
	{
		qsort(probs, n_vocab, sizeof(ProbIndex), cmp_prob_desc);
		/* Renormalize top-k */
		sum = 0.0f;
		for(int i = 0; i < top_k; i++)
			sum += probs[i].val;
		for(int i = 0; i < top_k; i++)
			probs[i].val /= sum;
		n_vocab = top_k;
	}

	/* Random sample */
	float r = (float)rand() / (float)RAND_MAX;
	float cumsum = 0.0f;
	for(int i = 0; i < n_vocab; i++)
	{
		cumsum += probs[i].val;
		if(cumsum >= r)
			return probs[i].idx;
	}
	return probs[n_vocab - 1].idx;
}

/* ── Main ────────────────────────────────────────────────────────────── */

static void usage(const char *prog)
{
	fprintf(stderr, "Usage: %s <model.gguf> [options]\n", prog);
	fprintf(stderr, "  -p <prompt>   Input prompt\n");
	fprintf(stderr, "  -n <int>      Max tokens to generate (default: 128)\n");
	fprintf(stderr, "  -t <float>    Temperature (default: 0.7)\n");
	fprintf(stderr, "  -c <int>      Max context length (default: 2048)\n");
	fprintf(stderr, "  -k <int>      Top-K sampling (default: 40)\n");
	fprintf(stderr, "  -s <int>      Random seed (default: time-based)\n");
	fprintf(stderr, "  --raw         Raw mode (no chat template)\n");
	fprintf(stderr, "  --no-think    Disable thinking mode\n");
}

int main(int argc, char **argv)
{
	if(argc < 2)
	{
		usage(argv[0]);
		return 1;
	}

	const char *model_path = argv[1];
	const char *prompt = "Hello, world!";
	int max_tokens = 128;
	float temperature = 0.7f;
	int max_ctx = 2048;
	int top_k = 40;
	unsigned int seed = (unsigned int)time(NULL);
	bool raw_mode = false;
	bool think_mode = true;

	for(int i = 2; i < argc; i++)
	{
		if(strcmp(argv[i], "-p") == 0 && i + 1 < argc)
			prompt = argv[++i];
		else if(strcmp(argv[i], "-n") == 0 && i + 1 < argc)
			max_tokens = atoi(argv[++i]);
		else if(strcmp(argv[i], "-t") == 0 && i + 1 < argc)
			temperature = atof(argv[++i]);
		else if(strcmp(argv[i], "-c") == 0 && i + 1 < argc)
			max_ctx = atoi(argv[++i]);
		else if(strcmp(argv[i], "-k") == 0 && i + 1 < argc)
			top_k = atoi(argv[++i]);
		else if(strcmp(argv[i], "-s") == 0 && i + 1 < argc)
			seed = atoi(argv[++i]);
		else if(strcmp(argv[i], "--raw") == 0)
			raw_mode = true;
		else if(strcmp(argv[i], "--no-think") == 0)
			think_mode = false;
		else
		{
			usage(argv[0]);
			return 1;
		}
	}
	srand(seed);

	/* Initialize IQ1S grid */
	init_iq1s_grid();

	/* Load model */
	Model *m = model_load(model_path);
	model_init_state(m, max_ctx);

	/* Tokenize prompt */
	int *prompt_tokens = malloc(max_ctx * sizeof(int));
	int n_prompt;
	if(raw_mode)
	{
		n_prompt = tokenize_raw(m->tok, prompt, prompt_tokens, max_ctx);
		printf("Prompt (raw): \"%s\" (%d tokens)\n\n", prompt, n_prompt);
	}
	else
	{
		n_prompt = tokenize_chat(m->tok, prompt, prompt_tokens, max_ctx, think_mode);
		printf("Prompt (chat%s): \"%s\" (%d tokens)\n\n", think_mode ? "+think" : "", prompt, n_prompt);
	}

	/* Generation loop */
	struct timespec t_start, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);

	int next_token = prompt_tokens[0];
	int pos = 0;
	int n_generated = 0;

	for(int step = 0; step < n_prompt + max_tokens - 1; step++)
	{
		float *logits = forward(m, next_token, pos);

		if(step < n_prompt - 1)
		{
			/* Prefill: feed next prompt token */
			next_token = prompt_tokens[step + 1];
		}
		else
		{
			/* Generate */
			if(step == n_prompt - 1)
			{
				/* First generation step after prefill — print marker */
				fflush(stdout);
			}
			next_token = sample(logits, m->n_vocab, temperature, top_k);
			if(next_token == m->tok->eos_id || next_token == TOK_END_TURN || next_token == 1 /* <eos> base */)
			{
				printf("\n[EOS]\n");
				break;
			}
			print_token(m->tok, next_token);
			fflush(stdout);
			n_generated++;
		}
		pos++;

		if(pos >= max_ctx)
		{
			printf("\n[Context limit reached]\n");
			break;
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &t_end);
	double elapsed = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

	printf("\n\n--- Stats ---\n");
	printf("Prompt tokens: %d\n", n_prompt);
	printf("Generated tokens: %d\n", n_generated);
	printf("Total time: %.2f s\n", elapsed);
	if(n_generated > 0)
		printf("Speed: %.2f tokens/s\n", (double)n_generated / elapsed);

	free(prompt_tokens);
	return 0;
}
