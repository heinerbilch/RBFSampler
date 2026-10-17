//
//  LinearSGDSVM.swift
//  MyTool
//
//  Created by Heiner Strauss on 15.10.26.
//
import MLX
import MLXRandom

// MARK: - Lineares SVM-Training (Hinge-Loss, L2) per Mini-Batch-SGD

/// A linear support vector machine (SVM) trained using stochastic gradient descent (SGD) with hinge loss and L2 regularization.
///
/// This struct implements a linear SVM classifier that uses mini-batch SGD to optimize the hinge loss with L2 regularization.
/// The model learns a decision boundary defined by a weight vector `w` and a bias term `b`.
///
/// - Parameters:
///   - w: The weight vector of shape 1 × m, where m is the number of features.
///   - b: The bias term of shape 1 × 1.
///   - lambda: The regularization parameter for L2 regularization.
///   - learningRate: The learning rate for the SGD optimizer.
///
/// - Note: The labels should be in {-1, +1}.
struct LinearSGDSVM {
    /// The weight vector of shape 1 × m, where m is the number of features.
    var w: MLXArray
    
    /// The bias term of shape 1 × 1.
    var b: MLXArray
    
    /// The regularization parameter for L2 regularization.
    let lambda: Float
    
    /// The learning rate for the SGD optimizer.
    var learningRate: Float
    
    /// Initializes a new `LinearSGDSVM` instance.
    ///
    /// - Parameters:
    ///   - m: The number of features.
    ///   - lambda: The regularization parameter for L2 regularization. Defaults to 1e-4.
    ///   - learningRate: The learning rate for the SGD optimizer. Defaults to 0.05.
    init(featureCount m: Int, lambda: Float = 1e-4, learningRate: Float = 0.05) {
        self.w = MLXArray.zeros([1, m])
        self.b = MLXArray.zeros([1, 1])
        self.lambda = lambda
        self.learningRate = learningRate
    }

    /// Trains the SVM model using mini-batch SGD.
    ///
    /// - Parameters:
    ///   - x: The feature matrix of shape n × m, where n is the number of samples.
    ///   - y: The label vector of shape n × 1, where the labels are in {-1, +1}.
    ///   - epochs: The number of training epochs. Defaults to 20.
    ///   - batchSize: The size of each mini-batch. Defaults to 256.
    mutating func fit(_ x: MLXArray, _ y: MLXArray,
                      epochs: Int = 20, batchSize: Int = 256) {
        let n = x.shape[0]

        for _ in 0..<epochs {
            // Zufällige Reihenfolge: Permutation als Index-Array,
            // Indizes selbst kommen vom CPU-RNG (deterministisch egal hier).
            var order = Array(0..<n)
            order.shuffle()
            let idx = MLXArray(order.map { Int32($0) })

            let xShuffled = x[idx]   // n×m (Advanced Indexing)
            let yShuffled = y[idx]  // n×1

            var start = 0
            while start < n {
                let end = min(start + batchSize, n)
                let rows = end - start

                // Slices sind Views – keine Kopien von Hand nötig.
                let xb = xShuffled[start..<end]          // rows×m
                let yb = yShuffled[start..<end]          // rows×1

                // scores = xb @ wᵀ + b          → rows×1
                let scores = matmul(xb, w.T) + b
                let margins = scores * yb               // y * score
                let active = (margins .< 1).asType(.float32)  // 1{margin<1}

                // grad_w = λw - (2/rows) * Σ active*y*xb
                let ay = active * yb                     // rows×1
                let dataTerm = matmul(ay.T, xb) * (2.0 / Float(rows))
                let gradW = w * lambda - dataTerm

                // grad_b = -(2/rows) * Σ active*y
                let gradB = -sum(ay) * (2.0 / Float(rows))

                w = w - gradW * learningRate
                b = b - gradB * learningRate
                start = end
            }
            // Pro Epoche auswerten: haelt den lazy Graph flach, sonst waechst
            // er ueber alle Iterationen und der Speicher laeuft voll.
            w.eval()
            b.eval()
        }
    }

    /// Computes the decision function f(x) = x·wᵀ + b for the input features.
    ///
    /// - Parameter x: The feature matrix of shape n × m, where n is the number of samples.
    /// - Returns: The decision function values of shape n × 1.
    func decisionFunction(_ x: MLXArray) -> MLXArray {
        return matmul(x, w.T) + b
    }

    /// Predicts the class labels for the input features.
    ///
    /// - Parameter x: The feature matrix of shape n × m, where n is the number of samples.
    /// - Returns: The predicted class labels of shape n × 1, where the labels are in {-1, +1}.
    func predict(_ x: MLXArray) -> MLXArray {
        return (decisionFunction(x) .>= 0).asType(.float32) * 2 - 1
    }
}
