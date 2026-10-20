//
//  TuneGamma.swift
//  RBFToolkit
//
//  Created by Heiner Strauss on 16.10.26.
//

import Foundation
import MLX

/// Gittersuche über gamma (das, was in sklearn GridSearchCV macht).
/// Grid breit genug wählen, dass der CV-Score an BEIDEN Enden abfällt
/// (links Unterfitting, rechts Kernel-Rauschen) – andernfalls verbreitern.
/// Ties werden nicht per "erster Treffer" gelöst, sondern über die
/// geometrische Mitte des Plateaus (robust gegen Randlagen).
/// epochs/batchSize/learningRate/lambda werden an crossValidate
/// durchgereicht: Tuning und finales Modell müssen mit demselben
/// Budget trainieren, sonst vergleicht das Grid untertrainierte
/// Modelle – also Rauschen.
public func tuneGamma(x: MLXArray, y: MLXArray, nComponents: Int,
               grid: [Float] = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
               folds: Int = 5, tolerance: Float = 0.002,
               epochs: Int = 20, batchSize: Int = 256,
               learningRate: Float = 0.05, lambda: Float = 1e-4)
    -> (gamma: Float, score: Float) {
    precondition(grid == grid.sorted(), "Grid muss aufsteigend sortiert sein")
    var results: [(gamma: Float, score: Float)] = []
    for g in grid {
        let s = crossValidate(x: x, y: y, gamma: g, nComponents: nComponents,
                              folds: folds, epochs: epochs, batchSize: batchSize,
                              learningRate: learningRate, lambda: lambda)
        print(String(format: "gamma=%.4f  CV-Genauigkeit=%.3f", g, s))
        results.append((g, s))
    }

    let maxScore = results.map { $0.score }.max()!

    // Zusammenhängende Plateaus: Kandidaten, die das Maximum (bis auf
    // Toleranz) erreichen UND im Grid nebeneinander liegen. Getrennte
    // Plateaus (Multiskalen-Daten) werden nicht vermischt – sonst läge
    // die "Mitte" im Tal dazwischen.
    var runs: [[Int]] = []
    for (i, r) in results.enumerated() where maxScore - r.score <= tolerance {
        if let last = runs.last, last.last == i - 1 {
            runs[runs.count - 1].append(i)
        } else {
            runs.append([i])
        }
    }

    // Breitstes Plateau gewinnen (log-Spannweite); bei Gleichstand
    // das mit dem höheren mittleren Score.
    let measured = runs.map { run -> (run: [Int], span: Float, mean: Float) in
        let lo = results[run.first!].gamma, hi = results[run.last!].gamma
        let mean = run.map { results[$0].score }.reduce(0, +) / Float(run.count)
        return (run, log2(hi / lo), mean)
    }
    let best = measured.max { a, b in
        a.span == b.span ? a.mean < b.mean : a.span < b.span
    }!

    let gLow = results[best.run.first!].gamma
    let gHigh = results[best.run.last!].gamma

    if runs.count > 1 {
        print("Hinweis: \(runs.count) getrennte Plateaus – Multiskalen-Struktur? " +
              "Es wird das breiteste gewählt.")
    }
    if gLow == grid.first! {
        print("Warnung: Score am linken Grid-Rand noch maximal – Grid nach unten verbreitern")
    }
    if gHigh == grid.last! {
        print("Warnung: Score am rechten Grid-Rand noch maximal – Grid nach oben verbreitern")
    }

    // Geometrische Mitte DES GEWÄHLTEN Plateaus: sqrt(low * high).
    // Liegt garantiert im stabilen Bereich, nicht am Abbruch.
    let chosen = (gLow * gHigh).squareRoot()
    print(String(format: "Plateau [%.4f … %.4f]  → gewähltes gamma: %.4f", gLow, gHigh, chosen))
    return (chosen, maxScore)
}
