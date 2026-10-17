//
//  CrossValidate.swift
//  MyTool
//
//  Created by Heiner Strauss on 16.10.26.
//


import MLX
// MARK: - Cross-Validierung (k-fold)


/// k-fold CV über die gesammte Pipeline (RBFSampler + Hinge-SGD-SVM).
/// Liefert die mittlere Validierungs-Genauigkeit über alle Folds.
/// Wichtig: der Sampler wird pro Fold neu erzeugt – W hängt zwar nur von
/// gamma/d ab (kein Leakage durch die Zufallsgewichte), aber der SVM wird
/// ausschließlich auf den Trainings-Folds gefittet.
func crossValidate(x: MLXArray, y: MLXArray,
                   gamma: Float, nComponents: Int,
                   folds: Int = 5, seed: UInt64 = 7,
                   epochs: Int = 20, batchSize: Int = 256,
                   learningRate: Float = 0.05, lambda: Float = 1e-4) -> Float {
    let n = x.shape[0]
    var rng = SeededRNG(seed: seed)
    var order = Array(0..<n)
    order.shuffle(using: &rng)

    let foldSize = n / folds
    var scores: [Float] = []

    for k in 0..<folds {
        let valStart = k * foldSize
        let valEnd = (k == folds - 1) ? n : (k + 1) * foldSize
        let valIdx = order[valStart..<valEnd]
        let trainIdx = order[0..<valStart] + order[valEnd..<n]

        let trainI = MLXArray(trainIdx.map { Int32($0) })
        let valI = MLXArray(valIdx.map { Int32($0) })

        let xTr = x[trainI], yTr = y[trainI]
        let xVa = x[valI],   yVa = y[valI]

        let sampler = RBFSamplerMLX(dimension: x.shape[1],
                                    nComponents: nComponents, gamma: gamma, seed: 42)
        let zTr = sampler.transform(xTr)

        var svm = LinearSGDSVM(featureCount: nComponents,
                               lambda: lambda, learningRate: learningRate)
        svm.fit(zTr, yTr, epochs: epochs, batchSize: batchSize)

        let pred = svm.predict(sampler.transform(xVa))
        pred.eval()
        let correct = (pred * yVa .> 0).sum().item(Int.self)
        scores.append(Float(correct) / Float(valIdx.count))
    }
    return scores.reduce(0, +) / Float(scores.count)
}
