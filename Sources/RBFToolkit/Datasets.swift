//
//  Datasets.swift
//  RBFToolkit
//
//  Created by Heiner Strauss on 18.10.26.
//
import MLX
import MLXRandom

// MARK: - Synthetische Testdaten

/// Zwei Gauß-Blobs: Standardnormalverteilung, per Zufall gelabelt und um
/// ±shift verschoben – jedes Experiment startet mit denselben Daten.
/// Liefert (x: n×d, y: n×1 mit Labels in {-1, +1}).
public func makeBlobs(n: Int, d: Int, shift: Float = 0.8, seed: UInt64 = 1234) -> (x: MLXArray, y: MLXArray) {
    MLXRandom.seed(seed)
    let x = MLXRandom.normal([n, d])
    let y = MLXRandom.uniform(low: 0.0, high: 1.0, [n, 1]) .< 0.5
    let ySigned = y.asType(.float32) * 2 - 1
    let xShifted = x + ySigned * shift   // Blobs um ±shift verschoben
    return (xShifted, ySigned)
}
