/*
Copyright 2021 Fixstars Corporation
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http ://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include "cublas_v2.h"


static constexpr int NUM_TRIALS = 11;


// CPU版行列積カーネル
void matmul_cpu(float *C, const float *A, const float *B, int n){
	for(int i = 0; i < n; ++i){
		for(int j = 0; j < n; ++j){
			C[i * n + j] = 0.0f;
			for(int k = 0; k < n; ++k){
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}


// GPU版処理時間計測
// NUM_TRIALS 回計測して中央値を求める
double matmul_gpu_benchmark(
	float *h_C, const float *h_A, const float *h_B, int n, cublasHandle_t handle)
{
	const float alpha = 1.0f, beta = 0.0f;

	// デバイスメモリの確保
	float *d_C = nullptr, *d_A = nullptr, *d_B = nullptr;
	cudaMalloc(&d_A, sizeof(float) * n * n);
	cudaMalloc(&d_B, sizeof(float) * n * n);
	cudaMalloc(&d_C, sizeof(float) * n * n);
	// 入力データの転送
	cublasSetMatrix(n, n, sizeof(float), h_A, n, d_A, n);
	cublasSetMatrix(n, n, sizeof(float), h_B, n, d_B, n);

	std::vector<double> durations(NUM_TRIALS);
	for(int i = 0; i < NUM_TRIALS; ++i){
		const auto begin = std::chrono::steady_clock::now();
		cublasSgemm(
			handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			n, n, n,
			&alpha,
			d_B, n,
			d_A, n,
			&beta,
			d_C, n);
		cudaDeviceSynchronize();  // GPUカーネルの終了を待つ
		const auto end = std::chrono::steady_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
		durations[i] = duration.count() * 1e-3;
	}

	// 出力データの転送
	cublasGetMatrix(n, n, sizeof(float), d_C, n, h_C, n);
	// デバイスメモリの開放
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// 中央値を求める
	std::sort(durations.begin(), durations.end());
	return durations[NUM_TRIALS / 2];
}


// 検算
bool validate(const float *expect, const float *actual, int n){
	bool valid = true;
	for(int i = 0; i < n * n; ++i){
		if(std::fabs(expect[i] - actual[i]) > 1e-3){
			std::cerr << "(" << i / n << ", " << i % n << "): " << expect[i] << " != " << actual[i] << std::endl;
			valid = false;
		}
	}
	return valid;
}


int main(int argc, char *argv[]){
	if(argc < 2){
		std::cerr << "Usage: " << argv[0] << " n" << std::endl;
		return 0;
	}

	cublasHandle_t handle;
	cublasCreate(&handle);

	const int n = atoi(argv[1]);
	std::cout << "n = " << n << std::endl;

	std::default_random_engine engine;
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	std::vector<float> A(n * n), B(n * n), cpu_C(n * n), gpu_C(n * n);
	for(int i = 0; i < n * n; ++i){
		A[i] = dist(engine);
		B[i] = dist(engine);
	}

	// CPU側の計算が遅いのでキャッシュする
	const std::string cache_name = "cache_" + std::to_string(n);
	std::ifstream cache_ifs(cache_name, std::ios::binary);
	if(cache_ifs){
		cache_ifs.read(reinterpret_cast<char*>(cpu_C.data()), sizeof(float) * n * n);
	}else{
		std::ofstream cache_ofs(cache_name);
		matmul_cpu(cpu_C.data(), A.data(), B.data(), n);
		cache_ofs.write(reinterpret_cast<const char*>(cpu_C.data()), sizeof(float) * n * n);
	}

	const auto gpu_duration =
		matmul_gpu_benchmark(gpu_C.data(), A.data(), B.data(), n, handle);
	std::cout << "GPU: " << gpu_duration << " [ms]" << std::endl;
	std::cout << "     " << 2.0 * n * n * n / gpu_duration * 1e-9 << " [TFLOPS]" << std::endl;

	const auto valid = validate(cpu_C.data(), gpu_C.data(), n);
	std::cout << "Validation: " << (valid ? "Success" : "Failed") << std::endl;

	cublasDestroy(handle);

	return 0;
}
