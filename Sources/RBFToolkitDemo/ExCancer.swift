//
//  ExCancer.swift
//  RBFToolkitDemo
//
//  Created by Heiner Strauss on 19.10.26.
//
import Foundation
import MLX
import RBFToolkit

// MARK: - Experiment: Cancer (echte Daten)

/// Erster echter ML-Datensatz: Breast Cancer Wisconsin (WDBC). Der Weg durch
/// die Pipeline ist derselbe wie bei den Blobs, aber diesmal mit allem, was
/// echte Daten brauchen: CSV-Lader, deterministischer 70/30-Split,
/// Standardisierung (fit auf Train, apply auf beide – kein Leakage) und
/// gamma-Tuning ausschließlich auf der Trainings-CV. Die Testmenge wird
/// genau einmal am Ende angefasst.
///
/// Kern-Vergleich: dieselbe lineare SVM einmal direkt auf den 30
/// standardisierten Merkmalen (ohne Kernel) und einmal auf RFF-Features.
/// WDBC ist fast linear trennbar – wenn RFF hier kaum mehr Genauigkeit
/// kauft, ist das selbst das Ergebnis: Der Kernel hilft nur, wo die
/// Grenze wirklich nichtlinear ist.
struct Cancer: Experiment {
    static let name = "cancer"
    static let about = "WDBC (echte Daten): CSV-Lader → Split → Standardisierung → gamma-Tuning → Test vs. lineare Baseline (Schritte: data tune run)"

    init() {}

    private static let dataPath = "data/wdbc.data"
    private static let gammaGrid: [Float] = [0.0001, 0.0003, 0.001, 0.003,
                                             0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

    private static let steps: [(name: String, about: String)] = [
        ("data", "CSV laden: Zeilen, Klassenverteilung, Merkmals-Skalen"),
        ("tune", "Split → Standardisierung → gamma-Tuning auf Trainings-CV"),
        ("run", "Kompletter Durchlauf: Baseline, RFF-Test, m-Sweep")
    ]

    func run(steps: [String]) throws {
        let toRun = steps.isEmpty ? Cancer.steps.map { $0.name } : steps
        for s in toRun { try runStep(s) }
    }

    private func runStep(_ step: String) throws {
        switch step {
        case "data":
            try runData()
        case "tune":
            try runTune()
        case "run":
            try runAll()
        default:
            print("Unbekannter Schritt: " + step + ". Verfügbare Schritte:")
            for s in Cancer.steps { print("  " + s.name + "  –  " + s.about) }
        }
    }

    /// WDBC laden, splitten, standardisieren (Statistiken nur aus Train).
    private func prepared() throws -> (sTrain: MLXArray, yTrain: MLXArray,
                                        sTest: MLXArray, yTest: MLXArray) {
        let (x, y) = try loadWDBC(path: Cancer.dataPath)
        let (xTrain, yTrain, xTest, yTest) = splitTrainTest(x: x, y: y, seed: 123)
        let std = Standardizer(x: xTrain)
        let sTrain = std.apply(xTrain)
        sTrain.eval()
        let sTest = std.apply(xTest)
        sTest.eval()
        return (sTrain, yTrain, sTest, yTest)
    }

    /// Schritt data: Zeilen, Klassen, und warum Standardisierung nötig ist.
    private func runData() throws {
        let (x, y) = try loadWDBC(path: Cancer.dataPath)
        let n = x.shape[0], d = x.shape[1]
        let yArr = y.asArray(Float.self)
        let malignant = yArr.filter { $0 > 0 }.count
        print("WDBC geladen: \(n) Punkte, \(d) Merkmale")
        print(String(format: "Klassenverteilung: %d maligne (+1), %d benigne (-1)",
                     malignant, n - malignant))

        let xArr = x.asArray(Float.self)
        for j in 0..<d {
            var lo: Float = .infinity, hi: Float = -.infinity
            for i in 0..<n {
                let v = xArr[i * d + j]
                lo = min(lo, v)
                hi = max(hi, v)
            }
            print(String(format: "  Merkmal %2d: [%10.4g … %10.4g]", j, lo, hi))
        }
        print("Die Skalen liegen teileweise um Zehnerpotenzen auseinander – ohne Standardisierung dominieren die großskaligen Spalten jede Distanz.")
    }

    /// Schritt tune: nur die Trainingshälfte, 5-fach-CV über das gamma-Grid.
    private func runTune() throws {
        let (sTrain, yTrain, _, _) = try prepared()
        print("Trainings-Split standardisiert (mean/std nur aus Train, Test unverarbeitet beiseite).")
        let best = tuneGamma(x: sTrain, y: yTrain, nComponents: 500, grid: Cancer.gammaGrid)
        print(String(format: "Gewähltes gamma: %.4f (CV-Score %.3f)", best.gamma, best.score))
    }

    /// Schritt run: alles end-to-end, Testmenge genau einmal angefasst.
    private func runAll() throws {
        let (sTrain, yTrain, sTest, yTest) = try prepared()
        let d = sTrain.shape[1]
        let nTest = sTest.shape[0]

        // 1) Lineare Baseline: dieselbe SVM direkt auf den 30 standardisierten
        //    Merkmalen, ohne Kernel-Trick. Der Vergleichspunkt: Was kauft RFF?
        var lin = LinearSGDSVM(featureCount: d, lambda: 1e-4, learningRate: 0.05)
        lin.fit(sTrain, yTrain, epochs: 20, batchSize: 256)
        let linPred = lin.predict(sTest)
        linPred.eval()
        let linAcc = 100.0 * Float((linPred * yTest .> 0).sum().item(Int.self)) / Float(nTest)
        print(String(format: "Lineare Baseline (kein Kernel): Test-Genauigkeit %.1f%%", linAcc))

        // 2) gamma-Tuning NUR auf der Trainings-CV
        let best = tuneGamma(x: sTrain, y: yTrain, nComponents: 500, grid: Cancer.gammaGrid)

        // 3) Finales RFF-Modell: Features von Train und Test durch denselben
        //    Sampler (W, b sind gamma/d-Parameter, kein Fit auf Daten).
        let sampler = RBFSamplerMLX(dimension: d, nComponents: 500,
                                    gamma: best.gamma, seed: 42)
        let zTrain = sampler.transform(sTrain)
        var svm = LinearSGDSVM(featureCount: 500, lambda: 1e-4, learningRate: 0.05)
        svm.fit(zTrain, yTrain, epochs: 20, batchSize: 256)
        let zTest = sampler.transform(sTest)
        let pred = svm.predict(zTest)
        pred.eval()
        let acc = 100.0 * Float((pred * yTest .> 0).sum().item(Int.self)) / Float(nTest)
        let dec = svm.decisionFunction(zTest)
        dec.eval()
        print(String(format: "RFF (m=500, gamma=%.4g): Test-Genauigkeit %.1f%%, f in [%.2f, %.2f]",
                     best.gamma, acc,
                     dec.min().item(Float.self), dec.max().item(Float.self)))

        // 4) m-Sweep: Näherungsqualität der Feature-Map auf echten Daten.
        print(String(format: "m-Sweep (gamma=%.4g):", best.gamma))
        for m in [100, 250, 500, 1000] {
            let s = RBFSamplerMLX(dimension: d, nComponents: m, gamma: best.gamma, seed: 42)
            let zTr = s.transform(sTrain)
            var sv = LinearSGDSVM(featureCount: m, lambda: 1e-4, learningRate: 0.05)
            sv.fit(zTr, yTrain, epochs: 20, batchSize: 256)
            let p = sv.predict(s.transform(sTest))
            p.eval()
            let a = 100.0 * Float((p * yTest .> 0).sum().item(Int.self)) / Float(nTest)
            print(String(format: "  m=%4d: Test-Genauigkeit %.1f%%", m, a))
        }

        print("Disziplin: gamma, Standardisierung und alle Zwischenzahlen kennen nur den Trainings-Split; der Test fließt ausschließlich in die finalen Prozentwerte ein.")
    }
}
