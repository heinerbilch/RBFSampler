//
//  RBFSamplerMLX.swift
//  MyTool
//
//  Created by Heiner Strauss on 15.10.26.
//

import Foundation
import MLX
import MLXRandom


// MARK: - RBFSampler (Random Fourier Features)

/// A struct that implements the Random Fourier Features (RFF) method for the RBF kernel.
///
/// The RBFSamplerMLX struct provides a way to approximate the RBF kernel using random Fourier features.
/// This approximation is useful for large-scale kernel methods, as it allows for efficient computation
/// of the kernel matrix without explicitly computing the pairwise distances between all data points.
///
/// The RBFSamplerMLX struct is initialized with the input dimension `d`, the number of components `m`,
/// and the gamma parameter `gamma` for the RBF kernel. The weight matrix `W` and the bias vector `b` are
/// initialized with values drawn from a normal distribution and a uniform distribution, respectively.
///
/// The `transform` method projects the input data into a higher-dimensional space using the weight matrix `W`,
/// adds the bias vector `b`, and applies the cosine function. The result is then scaled by the factor `sqrt(2/m)`,
/// where `m` is the number of components.
struct RBFSamplerMLX {
    let nComponents: Int
    let gamma: Float
    
    /// The weight matrix for the RBF sampler.
    /// This matrix is of size d × m, where d is the input dimension and m is the number of components.
    /// It is initialized with values drawn from a normal distribution with mean 0 and standard deviation sqrt(2 * gamma).
    /// The matrix is used to project the input data into a higher-dimensional space.
    let W: MLXArray   // d × m, ~ N(0, 2*gamma)
    let b: MLXArray   // 1 × m, ~ Uniform[0, 2π)
    
    /// Initializes a new RBFSamplerMLX instance.
    ///
    /// - Parameters:
    ///   - d: The input dimension.
    ///   - m: The number of components (features) to generate.
    ///   - gamma: The gamma parameter for the RBF kernel.
    ///   - seed: The seed for the random number generator. Defaults to 42.
    ///
    /// - Returns: An initialized RBFSamplerMLX instance.
    init(dimension d: Int, nComponents m: Int, gamma: Float, seed: UInt64 = 42) {
        MLXRandom.seed(UInt64(seed))
        self.nComponents = m
        self.gamma = gamma
        // MLX besitzt normal(uniform) direkt – kein eigener RNG nötig.
        self.W = MLXRandom.normal([d, m]) * pow(2 * gamma, 0.5)
        self.b = MLXRandom.uniform(low: 0.0, high: 2 * Float.pi, [1, m])
    }
    
    /// X: n×d → Z: n×m
    /// Transforms the input data using the RBF sampler.
    ///
    /// The transformation involves projecting the input data into a higher-dimensional space using the weight matrix `W`,
    /// adding the bias vector `b`, and applying the cosine function. The result is then scaled by the factor `sqrt(2/m)`,
    /// where `m` is the number of components.
    ///
    /// - Parameter x: The input data of shape `n × d`, where `n` is the number of samples and `d` is the input dimension.
    /// - Returns: The transformed data of shape `n × m`, where `m` is the number of components.
    func transform(_ x: MLXArray) -> MLXArray {
            let proj = matmul(x, W)          // n×m
            let features = cos(proj + b)     // broadcast über Zeilen
            return features * pow(2.0 / Float(nComponents), 0.5)
    }
}
