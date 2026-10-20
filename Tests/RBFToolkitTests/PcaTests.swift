//
//  PcaTests.swift
//  RBFToolkitTests
//
//  Created by Heiner Strauss on 20.10.26.
//
import XCTest
import MLX
import MLXRandom
@testable import RBFToolkit

/// PCA per Potenzmethode: auf synthetischen Daten mit bekannter
/// Varianzrichtung muessen die beiden dominanten Achsen samt Eigenwerte
/// rauskommen – Vorzeichen der Eigenvektoren ist frei, nur die Achse zaehlt.
final class PcaTests: XCTestCase {
    func testPcaTopTwoFindsVarianceAxes() {
        MLXRandom.seed(5)
        let n = 4000
        let a = MLXRandom.normal([n, 1]) * 2     // x-Richtung, Varianz ~4
        let b = MLXRandom.normal([n, 1]) * 0.5    // y-Richtung, Varianz ~0.25
        a.eval()
        b.eval()
        let aArr = a.asArray(Float.self)
        let bArr = b.asArray(Float.self)
        var pts: [Float] = []
        pts.reserveCapacity(n * 2)
        for i in 0..<n { pts.append(aArr[i]); pts.append(bArr[i]) }
        let x = MLXArray(pts).reshaped([n, 2])

        let (v1, v2, l1, l2, total) = pcaTopTwo(x: x)
        let v1Arr = v1.asArray(Float.self)
        let v2Arr = v2.asArray(Float.self)

        XCTAssertGreaterThan(abs(v1Arr[0]), 0.99, "PC1 soll die x-Achse sein (Varianz 4)")
        XCTAssertLessThan(abs(v1Arr[1]), 0.05)
        XCTAssertGreaterThan(abs(v2Arr[1]), 0.99, "PC2 soll die y-Achse sein (Varianz 0.25)")
        XCTAssertLessThan(abs(v2Arr[0]), 0.05)
        XCTAssertEqual(l1, 4.0, accuracy: 0.2)
        XCTAssertEqual(l2, 0.25, accuracy: 0.05)
        XCTAssertEqual(total, 4.25, accuracy: 0.2, "Gesamtvarianz = Spur der Kovarianz")

        // Orthogonalitaet nach Deflation
        let dot = v1Arr[0] * v2Arr[0] + v1Arr[1] * v2Arr[1]
        XCTAssertLessThan(abs(dot), 0.05, "v2 muss orthogonal zu v1 sein (Deflation)")
    }
}
