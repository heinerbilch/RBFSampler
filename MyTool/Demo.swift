//
//  Demo.swift
//  MyTool
//
//  Created by Heiner Strauss on 15.10.26.
//
import Foundation
import MLX
import MLXRandom

// MARK: - Demo

@main
enum Demo {
    static func main() {
        // 1) Synthetische Daten: zwei Gauß-Blobs
        MLXRandom.seed(1234)
        let n = 20_000, d = 32
        let x = MLXRandom.normal([n, d])
        let y = MLXRandom.uniform(low: 0.0, high: 1.0, [n, 1]) .< 0.5
        let ySigned = y.asType(.float32) * 2 - 1
        let xShifted = x + ySigned * 0.8   // Blobs um ±0.8 verschoben

        // 1b) Testmenge plotten (erste zwei Dimensionen, interaktives HTML)
        let plotfile = NSHomeDirectory() + "/Desktop/scatter.html"
        plotScatter(x: xShifted, label: ySigned, to: plotfile)

        // 2) RBF-Features (GPU)
        let t0 = Date()
//        let sampler = RBFSamplerMLX(dimension: d, nComponents: 500, gamma: 0.5, seed: 42)
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
        // 6) Genauigkeit des getunten Modells (in-sample) – mit zBest, nicht z
        let predBest = svm.predict(zBest)
        predBest.eval()
        let bestCorrect = (predBest * ySigned .> 0).sum().item(Int.self)
        print(String(format: "Trainingsgenauigkeit (getunt): %.1f%%",
                     100.0 * Float(bestCorrect) / Float(n)))
        // 7) Export für VisIt/ParaView: Punkte mit Label und Entscheidungswert
        //    Entscheidungswerte des GETUNTEN Modells → Features zBest verwenden
        let vtkfile = NSHomeDirectory() + "/Desktop/points.vtk"
        writeVtk(x: xShifted, label: ySigned, decision: svm.decisionFunction(zBest),
                 to: vtkfile)
    }
}
