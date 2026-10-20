//
//  PreprocessTests.swift
//  RBFToolkitTests
//
//  Created by Heiner Strauss on 19.10.26.
//
import XCTest
import MLX
@testable import RBFToolkit

/// Vorverarbeitung: Standardizer (mean 0, std 1), deterministicher Split,
/// WDBC-CSV-Lader (Parsing und Fehlerfälle).
final class PreprocessTests: XCTestCase {

    func testStandardizerCentersAndScales() {
        let (x, _) = makeBlobs(n: 2000, d: 5, shift: 3.0)
        let std = Standardizer(x: x)
        let s = std.apply(x)
        s.eval()
        let sArr = s.asArray(Float.self)
        let n = x.shape[0], d = 5

        var maxMeanErr: Float = 0
        var maxStdErr: Float = 0
        for j in 0..<d {
            var mu: Float = 0
            for i in 0..<n { mu += sArr[i * d + j] }
            mu /= Float(n)
            var sd: Float = 0
            for i in 0..<n { let t = sArr[i * d + j] - mu; sd += t * t }
            sd = (sd / Float(n)).squareRoot()
            maxMeanErr = max(maxMeanErr, abs(mu))
            maxStdErr = max(maxStdErr, abs(sd - 1))
        }
        XCTAssertLessThan(maxMeanErr, 1e-4, "Standardisierte Spalten sollen Mittelwert 0 haben")
        XCTAssertLessThan(maxStdErr, 1e-4, "Standardisierte Spalten sollen Standardabweichung 1 haben")
    }

    func testSplitIsDeterministic() {
        let (x, y) = makeBlobs(n: 1000, d: 4)
        let a = splitTrainTest(x: x, y: y, seed: 9)
        let b = splitTrainTest(x: x, y: y, seed: 9)
        XCTAssertEqual(a.xTrain.asArray(Float.self), b.xTrain.asArray(Float.self))
        XCTAssertEqual(a.xTest.asArray(Float.self), b.xTest.asArray(Float.self))
        XCTAssertEqual(a.yTest.asArray(Float.self), b.yTest.asArray(Float.self))
        XCTAssertEqual(a.xTrain.shape[0] + a.xTest.shape[0], 1000)
    }

    func testLoadWDBCParsesFixture() throws {
        let features = Array(repeating: "0.5", count: 30).joined(separator: ",")
        let csv = "1001,M," + features + "\n1002,B," + features + "\n\n"
        let path = NSTemporaryDirectory() + "wdbc_fixture.csv"
        try csv.write(toFile: path, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(atPath: path) }

        let (x, y) = try loadWDBC(path: path)
        XCTAssertEqual(x.shape, [2, 30])
        XCTAssertEqual(y.shape, [2, 1])
        let yArr = y.asArray(Float.self)
        XCTAssertEqual(yArr[0], 1.0, "M (maligne) muss +1 sein")
        XCTAssertEqual(yArr[1], -1.0, "B (benigne) muss -1 sein")
        XCTAssertEqual(x.asArray(Float.self)[0], 0.5)
    }

    func testLoadWDBCRejectsBadRows() throws {
        let path = NSTemporaryDirectory() + "wdbc_bad.csv"
        try "1,M,1,2\n".write(toFile: path, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(atPath: path) }

        XCTAssertThrowsError(try loadWDBC(path: path)) { error in
            XCTAssertTrue(error is DatasetError)
        }
    }
}
