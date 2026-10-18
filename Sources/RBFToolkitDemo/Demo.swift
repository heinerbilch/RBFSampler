//
//  Demo.swift
//  RBFToolkitDemo
//
//  Created by Heiner Strauss on 15.10.26.
//
import Foundation
import MLX
import MLXRandom
import RBFToolkit

// MARK: - Demo

@main
enum Demo {
    static func main() {
        let vtkpath = NSHomeDirectory() + "/Desktop/"
        // 1) Synthetische Daten: zwei Gauß-Blobs
        MLXRandom.seed(1234)
        let n = 20_000, d = 32
        let x = MLXRandom.normal([n, d])
        let y = MLXRandom.uniform(low: 0.0, high: 1.0, [n, 1]) .< 0.5
        let ySigned = y.asType(.float32) * 2 - 1
        let xShifted = x + ySigned * 0.8   // Blobs um ±0.8 verschoben

        // 2) RBF-Features (GPU)
        let t0 = Date()
        let sampler = RBFSamplerMLX(dimension: d, nComponents: 500,
                                    gamma: 1.0 / Float(d), seed: 42)
        let z = sampler.transform(xShifted)
        z.eval()
        print(String(format: "Feature-Map: %d×%d → %d×%d in %.3fs",
                     n, d, n, sampler.nComponents, -t0.timeIntervalSinceNow))

        // 3) Lineares SVM-Training (GPU)
        var svm = LinearSGDSVM(featureCount: sampler.nComponents,
                               lambda: 1e-4, learningRate: 0.05)
        let t1 = Date()
        svm.fit(z, ySigned, epochs: 20, batchSize: 256)
        print(String(format: "Training: %.3fs", -t1.timeIntervalSinceNow))

        // 4) Genauigkeit (in-sample Sanity-Check)
        let pred = svm.predict(z)
        pred.eval()
        let correct = (pred * ySigned .> 0).sum().item(Int.self)
        print(String(format: "Trainingsgenauigkeit: %.1f%%",
                     100.0 * Float(correct) / Float(n)))

        // 5) Korrektheits-Check der Feature-Map gegen CPU-Nachrechnung (Zeile 0)
        verifyFeatureMap(sampler: sampler, x: xShifted, z: z, d: d)

        // Das gleiche nochmal mit gamma Tuning
        let best = tuneGamma(x: xShifted, y: ySigned, nComponents: 500)
        // RBF
        let t2 = Date()
        let bestSampler = RBFSamplerMLX(dimension: d, nComponents: 500,
                                        gamma: best.gamma, seed: 42)
        let zBest = bestSampler.transform(xShifted)
        zBest.eval()
        print(String(format: "Feature-Map: %d×%d → %d×%d in %.3fs",
                     n, d, n, bestSampler.nComponents, -t2.timeIntervalSinceNow))
        // Linear SVM-Training
        svm = LinearSGDSVM(featureCount: bestSampler.nComponents, lambda: 1e-4, learningRate: 0.05)
        let t3 = Date()
        svm.fit(zBest, ySigned, epochs: 20, batchSize: 256)
        print(String(format: "Training: %.3fs", -t3.timeIntervalSinceNow))
        // 6) Genauigkeit des getunten Modells (in-sample)
        let predBest = svm.predict(zBest)
        predBest.eval()
        let bestCorrect = (predBest * ySigned .> 0).sum().item(Int.self)
        print(String(format: "Trainingsgenauigkeit (getunt): %.1f%%",
                     100.0 * Float(bestCorrect) / Float(n)))

        // 7) Export für VisIt/ParaView: Entscheidungswerte des GETUNTEN Modells,
        //    inkl. Min/Max für die Colormap-Einstellung in ParaView.
        let decTuned = svm.decisionFunction(zBest)
        decTuned.eval()
        print(String(format: "Entscheidungswerte (λ=1e-4): min=%.2f, max=%.2f",
                     decTuned.min().item(Float.self),
                     decTuned.max().item(Float.self)))
        writeVtk(x: xShifted, label: ySigned, decision: decTuned, to: vtkpath + "points.vtk")

        // 8) Lambda-Vergleich: gleiche Daten, gleiche Features, nur λ anders.
        //    Erwartung: gleiche Genauigkeit und Grenze; Beträge der Entscheidungswerte
        //    nur leicht kleiner. Das |w|-Gleichgewicht (|w| ∝ 1/√λ) stellt sich erst
        //    über viel längere Trainingshorizonte ein – und die RFF-Features liegen
        //    alle auf der Einheitssphäre (||z||≈1), was die Skala ohnehin deckelt.
        var svmReg = LinearSGDSVM(featureCount: bestSampler.nComponents,
                                  lambda: 1e-2, learningRate: 0.05)
        svmReg.fit(zBest, ySigned, epochs: 20, batchSize: 256)
        let predReg = svmReg.predict(zBest)
        predReg.eval()
        let accReg = 100.0 * Float((predReg * ySigned .> 0).sum().item(Int.self)) / Float(n)
        let decReg = svmReg.decisionFunction(zBest)
        decReg.eval()
        print(String(format: "λ=1e-2: Genauigkeit=%.1f%%, min=%.2f, max=%.2f",
                     accReg,
                     decReg.min().item(Float.self),
                     decReg.max().item(Float.self)))
        writeVtk(x: xShifted, label: ySigned, decision: decReg,
                 to: vtkpath + "points_lambda1e-2.vtk")

        // 9) RBFSampler verstehen: Wellen (gamma-Effekt), Wellenfeld (ParaView),
        //    und der Beweis, dass z(x)·z(x') den Gauß-Kernel rekonstruiert.
        plotRbfWaves(gamma: 0.05, to: vtkpath + "waves_small_gamma.html")
        plotRbfWaves(gamma: 2.0, to: vtkpath + "waves_large_gamma.html")
        writeVtkWaveField(gamma: 0.25, to: vtkpath + "wave.vtk")
        plotKernelApprox(x: xShifted, gamma: best.gamma, to: vtkpath + "kernel_approx.html")
    }
}
