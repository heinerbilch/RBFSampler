//
//  KernelApproxTests.swift
//  RBFToolkitTests
//
//  Created by Heiner Strauss on 17.10.26.
//
import XCTest
import MLX
import MLXRandom
@testable import RBFToolkit

/// Der Kern des RFF-Verfahrens als Test: z(x)·z(x') muss den Gauß-Kernel
/// e^{-gamma||x-x'||^2} approximieren (Fehler ~ Monte Carlo, fällt mit 1/sqrt(m)).
final class KernelApproxTests: XCTestCase {
    func testDotProductApproximatesGaussianKernel() {
        MLXRandom.seed(5)
        let d = 8, n = 64, m = 2000
        let gamma: Float = 0.1
        let x = MLXRandom.normal([n, d])
        let sampler = RBFSamplerMLX(dimension: d, nComponents: m, gamma: gamma, seed: 42)
        let z = sampler.transform(x)
        z.eval()

        let xArr = x.asArray(Float.self)
        let zArr = z.asArray(Float.self)

        var rng = SeededRNG(seed: 99)
        let pairs = 200
        var sumErr: Float = 0, maxErr: Float = 0
        for _ in 0..<pairs {
            let i = Int(rng.next() % UInt64(n))
            let j = Int(rng.next() % UInt64(n))
            var dist2: Float = 0, dot: Float = 0
            for t in 0..<d {
                let dv = xArr[i * d + t] - xArr[j * d + t]
                dist2 += dv * dv
            }
            for k in 0..<m { dot += zArr[i * m + k] * zArr[j * m + k] }
            let kernel = exp(-gamma * dist2)
            let err = abs(kernel - dot)
            maxErr = max(maxErr, err)
            sumErr += err
        }
        XCTAssertLessThan(sumErr / Float(pairs), 0.05, "mittlerer Kernel-Fehler zu groß")
        XCTAssertLessThan(maxErr, 0.2, "maximaler Kernel-Fehler zu groß")
    }

    /// Alle RFF-Feature-Vektoren liegen näherungsweise auf der Einheitssphäre
    /// (E[||z||^2] = 1, da z_k = sqrt(2/m) cos(...) und E[cos^2] = 1/2).
    func testFeatureVectorsNearUnitSphere() {
        MLXRandom.seed(9)
        let d = 8, n = 32, m = 2000
        let x = MLXRandom.normal([n, d])
        let sampler = RBFSamplerMLX(dimension: d, nComponents: m, gamma: 0.1, seed: 42)
        let z = sampler.transform(x)
        z.eval()
        let zArr = z.asArray(Float.self)

        for i in 0..<n {
            var norm2: Float = 0
            for k in 0..<m { norm2 += zArr[i * m + k] * zArr[i * m + k] }
            XCTAssertGreaterThan(norm2, 0.9, "Feature-Vektor \(i): Norm unter Einheitssphäre")
            XCTAssertLessThan(norm2, 1.1, "Feature-Vektor \(i): Norm über Einheitssphäre")
        }
    }
}
