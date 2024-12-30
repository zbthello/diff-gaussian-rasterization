/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// 正向传播，用于将每个高斯的输入球面谐波系数转换为简单的RGB颜色。
// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
    // 该实现基于Zhang等人（2022）的“基于可微分点的辐射场高效视图合成代码”
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir); //方向

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
    // RGB 颜色被限制为正值。如果值被限制，我们需要跟踪反向传播的情况。
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

/**
 * 预处理和投影（preprocessCUDA）
  计算投影圆圈的半径：在3D空间中的高斯分布投影到2D图像平面时，
  它通常会形成一个圆圈（实际上是椭圆，因为视角的影响）。这个步骤涉及计算这个圆圈的半径。
  计算圆圈覆盖的像素数：这涉及到将图像平面分成许多小块（tiles），
  并计算每个高斯分布投影形成的圆圈与哪些小块相交。这是为了高效地渲染，只更新受影响的小块。
 */
 /**
  * 预处理函数，作用是：
  * 检查Gaussian是否可见（在视锥内）；
  * 计算3维、2维协方差矩阵；
  * 把Gaussian投影到图像上，计算图像上的中心坐标、半径和覆盖的tile；
  * 计算颜色、深度等杂项。
  */
// 在光栅化之前，对每个高斯函数执行初始步骤。
// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(
        int P, // Gaussian的数量
        int D, // 对应于GaussianModel.active_sh_degree，是球谐度数
        int M, // RGB三通道的球谐傅里叶系数个数，应等于3 × (D + 1)²
	const float* orig_points, // Gaussian中心位置
	const glm::vec3* scales, // 缩放
	const float scale_modifier, // 缩放的修正项
	const glm::vec4* rotations, // 旋转
	const float* opacities, // 不透明度
	const float* shs, // 球谐系数
	bool* clamped, //表示每个值是否被截断了（RGB只能为正数），这个在反向传播的时候用
	const float* cov3D_precomp, // 预先计算的3维协方差矩阵
	const float* colors_precomp, // 预先计算的RGB颜色
	const float* viewmatrix, // W2C矩阵
	const float* projmatrix, // 投影矩阵
	const glm::vec3* cam_pos, // 相机坐标
	const int W, int H, // 图片宽高
	const float tan_fovx, float tan_fovy, // 视场角一半的正切值
	const float focal_x, float focal_y, //x,y方向的焦距
	int* radii, // Gaussian在像平面坐标系下的半径
	float2* points_xy_image, // Gaussian中心在图像上的像素坐标
	float* depths, // Gaussian中心的深度，即其在相机坐标系的z轴的坐标
	float* cov3Ds,  // 三维协方差矩阵
	float* rgb, // 根据球谐算出的RGB颜色值
	float4* conic_opacity, // 椭圆对应二次型的矩阵和不透明度的打包存储
	const dim3 grid, // tile的在x、y方向上的数量
	uint32_t* tiles_touched, // Gaussian覆盖的tile数量
	bool prefiltered)
{
    // 该函数预处理第idx个Gaussian
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    // 将半径和触摸的图块初始化为0。如果不改变这一点，这个高斯函数将不会被进一步处理。
	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

    // 进行近距离选择，如果在外面退出。
	// Perform near culling, quit if outside.
	float3 p_view; // Gaussian中心在相机坐标系下的坐标
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return; // 不在相机的视锥内就不管了

    // 通过投影转换点
	// Transform point by projecting
    // 3 * idx 这个表达式用于从一维数组 orig_points 中索引一个3D点的坐标。
    // 这里的 3 表示一个3D点由三个坐标组成：x、y 和 z。
    // 由于 orig_points 是一个一维数组，它按顺序存储了所有点的坐标，
    // 因此需要通过乘以 3 来计算出特定点的坐标在数组中的位置。
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix); // homogeneous coordinates（齐次坐标）
	float p_w = 1.0f / (p_hom.w + 0.0000001f); // 想要除以p_hom.w从而转成正常的3D坐标，这里防止除零
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    // 如果预先计算了3D协方差矩阵，则使用它，否则根据缩放和旋转参数进行计算。
	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters.
    // 因为3x3矩阵有6个元素[三角]，所以每个协方差矩阵占据连续的6个内存位置
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

    // 计算二维屏幕空间协方差矩阵
	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

    // 逆协方差（EWA算法）
	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y); // 二维协方差矩阵的行列式
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det; // 行列式的逆
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
    // conic是cone的形容词，意为“圆锥的”。猜测这里是指圆锥曲线（椭圆）。
    // 二阶矩阵求逆口诀：“主对调，副相反”。
    // conic 似乎是一个向量，用来存储与高斯分布相关的逆协方差矩阵的元素，
    // 这与圆锥曲线没有直接关系。

    // 计算屏幕空间中的范围（通过找到2D协方差矩阵的特征值）。
    // 使用范围来计算与此高斯重叠的屏幕空间块的边界矩形。如果矩形覆盖0个图块，则退出。
	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    // 韦达定理求二维协方差矩阵的特征值(特征值为椭圆的长短轴，将长轴近似为圆的半径)
    // 这里就是截取Gaussian的中心部位（3σ原则），只取像平面上半径为my_radius的部分
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
    // 检查该Gaussian在图片上覆盖了哪些tile（由一个tile组成的矩形表示）
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return; // 不与任何tile相交，不管了

    // 如果颜色已经预先计算，请使用它们，否则将球面谐波系数转换为RGB颜色。
	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

    // 为下一步存储一些有用的辅助数据。
	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z; // 深度，即相机坐标系的z轴
	radii[idx] = my_radius; // Gaussian在像平面坐标系下的半径
	points_xy_image[idx] = point_image; // Gaussian中心在图像上的像素坐标
    // 逆二维协方差和不透明度整齐地打包成一个float4
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
    // Gaussian中心在图像上的像素坐标
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// 主要光栅化方法。每个块协作处理一个图块，每个线程处理一个像素。
// 在获取数据和光栅化数据之间交替。“在获取数据和光栅化数据之间交替”
// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// 线程在读取数据（把数据从公用显存拉到block自己的显存）和进行计算之间来回切换，
// 使得线程们可以共同读取Gaussian数据。
// 这样做的原因是block共享内存比公共显存快得多
template <uint32_t CHANNELS> // CHANNELS取3，即RGB三个通道
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges, // 每个tile对应排过序的数组中的哪一部分
	const uint32_t* __restrict__ point_list, // 按tile、深度排序后的Gaussian ID列表
	int W, int H, // 图像宽高
	const float2* __restrict__ points_xy_image, // 图像上每个Gaussian中心的2D坐标
	const float* __restrict__ features, // RGB颜色
	const float4* __restrict__ conic_opacity, // 椭圆二次型的矩阵和不透明度的打包向量
	float* __restrict__ final_T, // 最终的透光率
	uint32_t* __restrict__ n_contrib, // 多少个Gaussian对该像素的颜色有贡献（用于反向传播时判断各个Gaussian有没有梯度）
	const float* __restrict__ bg_color, // 背景颜色
	float* __restrict__ out_color) // 渲染结果（图片）
{
    // 识别当前图块和相关的最小/最大像素范围。
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();  // block: 获取当前线程块的信息。
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X; // x方向上tile的个数 horizontal_blocks: 计算水平方向上的瓦片数量。
    // 负责的tile的坐标较小的那个角的坐标 pix_min和pix_max: 计算当前瓦片的最小和最大像素坐标。
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    // 负责的tile的坐标较大的那个角的坐标
    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    // 负责哪个像素 pix: 计算当前线程处理的像素坐标。
    uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    // 负责的像素在整张图片中的索引 pix_id: 计算当前像素的全局索引
    uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y }; // pix的浮点数版本

    // inside: 检查当前像素是否在图像范围内。
    // done: 如果像素在图像外，则设置done为true，这些线程将不参与光栅化。
	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
    // 已完成的线程可以帮助获取，但不会光栅化
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

    // 在位排序列表中加载要处理的ID的开始/结束范围。 range: 获取当前瓦片需要处理的高斯索引范围。
	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // BLOCK_SIZE = 16 * 16 = 256
    // 把任务分成rounds批，每批处理BLOCK_SIZE个Gaussians rounds: 计算需要处理的批次数量
    // 每一批，每个线程负责读取一个Gaussian的信息，
    // 所以该block的256个线程每一批就可以读取256个Gaussian的信息
    // _toDo: 计算当前批次需要处理的高斯数量。
    int toDo = range.y - range.x; // 要处理的Gaussian个数

    // 为批量集体提取的数据分配存储空间。分配共享内存用于存储每个批次中高斯的索引、位置和不透明度信息。
	// Allocate storage for batches of collectively fetched data.
    // __shared__: 同一block中的线程共享的内存
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
    // 初始化辅助变量 初始化透明度T、贡献计数器contributor、最后一次贡献记录last_contributor和颜色累加器C
	float T = 1.0f; // T = transmittance，透光率
	uint32_t contributor = 0; // 多少个Gaussian对该像素的颜色有贡献
	uint32_t last_contributor = 0; // 最后一次贡献记录
	float C[CHANNELS] = { 0 }; // 渲染结果

    // 迭代批次，直到全部完成或范围完成
    // 遍历每个批次，直到所有线程完成或范围完成。
    // 集体从全局内存获取高斯数据到共享内存。
    // 对于共享内存中的每个高斯，计算其对当前像素的贡献。
    // 使用高斯的锥形矩阵（conic_opacity）和位置（points_xy_image）来计算贡献。
    // 更新透明度（T）和颜色累加器（C）。
	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
        // 如果整个区块投票决定光栅化完成，则结束
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
        // 它首先具有__syncthreads的功能（让所有线程回到同一个起跑线上），
        // 并且返回对于多少个线程来说done是true。
		if (num_done == BLOCK_SIZE)
			break;

        // 将每个高斯数据从全局集中提取到共享
		// Collectively fetch per-Gaussian data from global to shared
        // Collectively fetch per-Gaussian data from global to shared
        // 由于当前block的线程要处理同一个tile，所以它们面对的Gaussians也是相同的
        // 因此合作读取BLOCK_SIZE条的数据。
        // 之所以分批而不是一次读完可能是因为block的共享内存空间有限
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
            // 读取负责的Gaussian信息
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();
        // block.sync()用于同步当前线程块中的所有线程。在 CUDA 中，线程块内的线程可以并行执行，
        // block.sync() 确保在继续执行后续代码之前，块内的所有线程都到达了这个点，即所有线程都完成了它们当前的工作。

        // 迭代当前批处理
		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
            // 跟踪范围内的当前位置
			// Keep track of current position in range
			contributor++;

            // 下面计算当前Gaussian的不透明度
            // 使用二次曲线矩阵重新取样（参见Zwicker等人的“表面飞溅”，2001）
			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j]; // Gaussian中心
			float2 d = { xy.x - pixf.x, xy.y - pixf.y }; // 该像素到Gaussian中心的位移向量
			float4 con_o = collected_conic_opacity[j];
            // 二维高斯分布公式的指数部分（见补充说明）（power）
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

            // 方程式（2）来自3D高斯溅射。
            // 通过乘以高斯不透明度及其从平均值的指数衰减来获得α。
            // 避免数值不稳定（见论文附录）。
			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power)); // Gaussian对于这个像素点来说的不透明度
            // 注意con_o.w是”opacity“，是Gaussian整体的不透明度
            if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

             // 方程式（3）来自3D高斯溅射。
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

            // 跟踪最后一个范围条目以更新此像素。
			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

    // 所有处理有效像素的线程都将其最终渲染数据写入帧和辅助缓冲区。
	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}0
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
    // 一个线程负责一个像素，一个block负责一个tile
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}