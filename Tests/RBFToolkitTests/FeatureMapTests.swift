//
//  FeatureMapTests.swift
//  RBFToolkitTests
//
//  Created by Heiner Strauss on 17.10.26.
//
import XCTest
import MLX
import MLXRandom
@testable import RBFToolkit

/// Korrektheit der MLX-Feature-Map gegen eine CPU-Nachrechnung
/// (das, was verifyFeatureMap in der Demo druckt, als harte Assertion).
final class FeatureMapTests: XCTestCase {
    func testTransformRowMatchesCPU() {
        MLXRandom.seed(123)
        let d = 8, m = 64
        let x = MLXRandom.normal([16, d])
        let sampler = RBFSamplerMLX(dimension: d, nComponents: m, gamma: 0.25, seed: 7)
        let z = sampler.transform(x)
        z.eval()

        let xArr = x.asArray(Float.self)
        let wArr = sampler.W.asArray(Float.self)
        let bArr = sampler.b.asArray(Float.self)
        let zArr = z.asArray(Float.self)
        let scale = Float(sqrt(2.0 / Double(m)))

        var maxErr: Float = 0
        for row in 0..<16 {
            for k in 0..<m {
                var s: Float = 0
                for j in 0..<d { s += xArr[row * d + j] * wArr[j * m + k] }
                let ref = scale * cos(s + bArr[k])
                maxErr = max(maxErr, abs(ref - zArr[row * m + k]))
            }
        }
        XCTAssertLessThan(maxErr, 1e-5, "Feature-Map weicht von CPU-Referenz ab")
    }
}
