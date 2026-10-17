//
//  ReproTests.swift
//  RBFToolkitTests
//
//  Created by Heiner Strauss on 17.10.26.
//
import XCTest
import MLX
import MLXRandom
@testable import RBFToolkit

/// Reproduzierbarkeit: gleiche Seeds müssen identische Modelle liefern.
final class ReproTests: XCTestCase {
    func testSameSeedGivesIdenticalModel() {
        MLXRandom.seed(3)
        let d = 16, m = 32
        let x = MLXRandom.normal([512, d])
        let y = MLXRandom.uniform(low: 0.0, high: 1.0, [512, 1]) .< 0.5
        let ySigned = y.asType(.float32) * 2 - 1

        let sampler = RBFSamplerMLX(dimension: d, nComponents: m, gamma: 0.05, seed: 11)
        let z = sampler.transform(x)
        z.eval()

        var svm1 = LinearSGDSVM(featureCount: m)
        var svm2 = LinearSGDSVM(featureCount: m)
        svm1.fit(z, ySigned, epochs: 3, batchSize: 64, seed: 42)
        svm2.fit(z, ySigned, epochs: 3, batchSize: 64, seed: 42)

        XCTAssertEqual(svm1.w.asArray(Float.self), svm2.w.asArray(Float.self))
        XCTAssertEqual(svm1.b.asArray(Float.self), svm2.b.asArray(Float.self))
        XCTAssertEqual(svm1.decisionFunction(z).asArray(Float.self),
                       svm2.decisionFunction(z).asArray(Float.self))
    }

    func testSeededRNGSequenceIsDeterministic() {
        var a = SeededRNG(seed: 7)
        var b = SeededRNG(seed: 7)
        for _ in 0..<100 {
            XCTAssertEqual(a.next(), b.next())
        }
    }
}
