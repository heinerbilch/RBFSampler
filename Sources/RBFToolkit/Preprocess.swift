//
//  Preprocess.swift
//  RBFToolkit
//
//  Created by Heiner Strauss on 19.10.26.
//
import Foundation
import MLX

// MARK: - Vorverarbeitung für echte Daten: Standardisierung + Split

/// Z-Standardisierung (Mittelwert 0, Standardabweichung 1) mit getrennten
/// fit/apply-Schritten – der entscheidende Punkt gegen Leakage: mean und
/// std werden NUR aus den Trainingsdaten berechnet, angewendet wird auf
/// beide Splits. Ohne Standardisierung dominieren großskalige Merkmale
/// (bei WDBC z.B. Flächengrößen 250–2500) die Distanzen im RBF-Kernel
/// komplett; gamma als "Wellenlänge" ist nur auf gleich skalierten Achsen
/// interpretierbar.
///
/// Die Spalten-Mittelwerte stehen hier als ones-vektor mal Matrix:
/// 1×n @ n×d = 1×d ist die Spaltensumme – dieselbe Matrizenlogik wie im
/// SVM-Training, nur diesmal für Statistik.
public struct Standardizer {
    /// Spaltenmittelwerte (Shape 1×d), nur aus den Trainingsdaten.
    public let mean: MLXArray

    /// Spalten-Standardabweichungen (Shape 1×d), nur aus den Trainingsdaten.
    public let std: MLXArray

    /// Berechnet mean und std pro Spalte aus x (n×d).
    /// NUR mit Trainingsdaten aufrufen – sonst leaked der Test-Split.
    public init(x: MLXArray) {
        let n = x.shape[0]
        let ones = MLXArray.ones([1, n])
        let mu = matmul(ones, x) / Float(n)                    // 1×d
        let diff = x - mu                                     // n×d (Broadcast)
        let varr = matmul(ones, diff * diff) / Float(n)       // 1×d
        mean = mu
        std = sqrt(varr + 1e-12)   // +1e-12: Nulldispersion kann nicht durch 0 teilen
    }

    /// Wendet (x - mean) / std an (Broadcast 1×d auf n×d).
    public func apply(_ x: MLXArray) -> MLXArray {
        return (x - mean) / std
    }
}

/// Deterministischer Train/Test-Split per SeededRNG (gleicher Seed =
/// bit-identischer Split). Die Testhälfte wird im weiteren Verlauf
/// ignoriert: gamma-Tuning und Standardisierung sehen nur die
/// Trainingshälfte, der Test wird genau einmal am Ende angefasst.
public func splitTrainTest(x: MLXArray, y: MLXArray, testFraction: Float = 0.3,
                           seed: UInt64 = 123)
    -> (xTrain: MLXArray, yTrain: MLXArray, xTest: MLXArray, yTest: MLXArray) {
    let n = x.shape[0]
    var rng = SeededRNG(seed: seed)
    var order = Array(0..<n)
    order.shuffle(using: &rng)
    let nTest = Int(Float(n) * testFraction)

    let testIdx = MLXArray(order[0..<nTest].map { Int32($0) })
    let trainIdx = MLXArray(order[nTest..<n].map { Int32($0) })

    // Advanced Indexing wie im Batch-Shuffling des SVM-Trainings
    return (x[trainIdx], y[trainIdx], x[testIdx], y[testIdx])
}
