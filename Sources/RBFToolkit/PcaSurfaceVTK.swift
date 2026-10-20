//
//  PcaSurfaceVTK.swift
//  RBFToolkit
//
//  Created by Heiner Strauss on 20.10.26.
//
import Foundation
import MLX

// MARK: - PCA (Potenzmethode) + WDBC-Entscheidungsoberflaeche in der PCA-Ebene

/// Die zwei dominanten Richtungen (Hauptachsen, PCA) per Potenzmethode von
/// Hand: Kovarianz C = X^T X / n als Matrixprodukt, dann wiederholtes C@v
/// mit Normierung – fuer die zweite Achse mit Deflation (C - λ1·v1·v1^T).
/// Keine linalg-Abhaengigkeit, dafuer Matrizenlogik wie im Rest des
/// Projekts und voll deterministisch (feste Iterationszahl, fester Start).
/// NUR auf Trainingsdaten aufrufen – auch die Leinwand einer
/// Visualisierung darf nicht aus Testpunkten gebaut werden.
///
/// - Returns: v1, v2 als d×1-Spalten (Vorzeichen ist frei, nur die Achse
///   zaehlt), die zugehoerigen Eigenwerte (Varianz entlang der Achse) und
///   die Gesamtvarianz (Spur von C) – damit laesst sich der erklaerte
///   Varianzanteil ausrechnen.
public func pcaTopTwo(x: MLXArray, iterations: Int = 200)
    -> (v1: MLXArray, v2: MLXArray, lambda1: Float, lambda2: Float, totalVariance: Float) {
    let n = x.shape[0]
    let d = x.shape[1]
    let C = matmul(x.T, x) / Float(n)   // d×d

    func dominant(_ M: MLXArray) -> (v: MLXArray, l: Float) {
        var v = MLXArray.ones([M.shape[0], 1]) / Float(M.shape[0]).squareRoot()
        var l: Float = 0
        for _ in 0..<iterations {
            let w = matmul(M, v)
            let nrm = sqrt(sum(w * w))
            l = nrm.item(Float.self)
            v = w / nrm
        }
        v.eval()
        return (v, l)
    }

    let (v1, l1) = dominant(C)
    let C2 = C - matmul(v1, v1.T) * l1        // Deflation: v1 rausrechnen
    let (v2, l2) = dominant(C2)

    let cArr = C.asArray(Float.self)
    var trace: Float = 0
    for i in 0..<d { trace += cArr[i * d + i] }
    return (v1, v2, l1, l2, trace)
}

/// Der WDBC-Datensatz hat 30 Merkmale – es gibt keine direkt zeichnbare
/// Entscheidungsoberflaeche wie bei den 2D-Blobs. Der Standardtrick:
/// die zwei Hauptachsen der Trainingsdaten als Leinwand. Das Modell bleibt
/// das echte 30-dimensionale RFF-Modell; die Flaeche ist der Schnitt durch
/// den Merkmalsraum entlang der von v1, v2 aufgespannten Ebene (alle
/// uebrigen 28 Koordinaten 0 – bei standardisierten Daten liegt der
/// Schwerpunkt im Ursprung, die Ebene geht also durch die "Mitte" der
/// Daten). Das Training ist dasselbe wie im cancer-Schritt run (gleiche
/// Seeds, gleiches Budget) – gleiches Modell, bit-identisch.
///
/// Zwei Dateien wie im surface-Schritt:
/// - Flaeche: STRUCTURED_GRID ueber der PCA-Ebene, z = f(x)·height, Skalar f.
/// - Punkte: POLYDATA, Trainings- UND Testpunkte auf Hoehe ihrer eigenen
///   Entscheidungswerte; Skalare label, decision und split (1=Train, 0=Test).
///
/// ParaView-Rezept:
/// - Flaeche nach 'f' faerben (diverging): maligne +1 als Berg, benigne -1
///   als Tal, Grenze am Farbwechsel; Contour f=0 zeichnet die Grenzkurve.
/// - Threshold -1.1 <= f <= 1.1 -> die Support-Vektor-Schale, hier auf
///   echten Krebsdaten.
/// - Punkte dazuladen, Glyph (Sphere), Color by 'label'; nach 'split'
///   filtern (Threshold), um Trainings- und Testpunkte zu trennen – die
///   Testpunkte sind die, aus denen die 97.1% bestehen.
/// - Realistisch bleiben: PC1/PC2 erklaeren nur ein Bruchteil der Varianz
///   (wird mitgedruckt) – die Ebene ist ein Schatten des 30D-Raums.
///   Punkte im "falschen" Tal sind meist in den 28 unsichtbaren
///   Koordinaten sauber getrennt; die Flaeche zeigt den Schnitt, nicht
///   das ganze Urteil des Modells.
public func writeVtkWDBCSurface(gamma: Float = 0.01,
                                nComponents: Int = 500,
                                epochs: Int = 300, batchSize: Int = 32,
                                resolution: Int = 100, height: Float = 0.5,
                                margin: Float = 0.1,
                                dataPath: String = "data/wdbc.data",
                                toSurface surfacePath: String = "wdbc_surface.vtk",
                                toPoints pointsPath: String = "wdbc_points.vtk") throws {
    // 1) Pipeline wie im cancer-Schritt run: laden, splitten, standardisieren,
    //    RFF-Modell trainieren (gleiche Seeds und Budget = gleiches Modell).
    let (x, y) = try loadWDBC(path: dataPath)
    let (xTrain, yTrain, xTest, yTest) = splitTrainTest(x: x, y: y, seed: 123)
    let nTrain = xTrain.shape[0]
    let nTest = xTest.shape[0]
    let std = Standardizer(x: xTrain)
    let sTrain = std.apply(xTrain)
    sTrain.eval()
    let sTest = std.apply(xTest)
    sTest.eval()
    let d = xTrain.shape[1]

    let sampler = RBFSamplerMLX(dimension: d, nComponents: nComponents,
                                gamma: gamma, seed: 42)
    let zTrain = sampler.transform(sTrain)
    var svm = LinearSGDSVM(featureCount: nComponents, lambda: 1e-4, learningRate: 0.05)
    svm.fit(zTrain, yTrain, epochs: epochs, batchSize: batchSize)
    let zTest = sampler.transform(sTest)
    let pred = svm.predict(zTest)
    pred.eval()
    let acc = 100.0 * Float((pred * yTest .> 0).sum().item(Int.self)) / Float(nTest)
    let decTrain = svm.decisionFunction(zTrain)
    decTrain.eval()
    let decTest = svm.decisionFunction(zTest)
    decTest.eval()
    print(String(format: "WDBC-Modell (gamma=%.4g): Test-Genauigkeit %.1f%%, f Train in [%.2f, %.2f], f Test in [%.2f, %.2f]",
                 gamma, acc,
                 decTrain.min().item(Float.self), decTrain.max().item(Float.self),
                 decTest.min().item(Float.self), decTest.max().item(Float.self)))

    // 2) PCA-Ebene, nur aus Trainingsdaten (auch hier: kein Leakage).
    let (v1, v2, l1, l2, totalVar) = pcaTopTwo(x: sTrain)
    print(String(format: "PCA: PC1 erklaert %.1f%%, PC2 %.1f%% der Varianz – die Ebene ist ein Schatten des 30D-Raums.",
                 100.0 * l1 / totalVar, 100.0 * l2 / totalVar))

    let v1Arr = v1.asArray(Float.self)
    let v2Arr = v2.asArray(Float.self)
    var vBoth: [Float] = []
    for i in 0..<d { vBoth.append(v1Arr[i]); vBoth.append(v2Arr[i]) }
    let V = MLXArray(vBoth).reshaped([d, 2])   // Spalten: v1, v2

    // 3) Alle Punkte in die Ebene projizieren (Train mit Train-Basis, Test auch).
    let pTrain = matmul(sTrain, V)
    pTrain.eval()
    let pTest = matmul(sTest, V)
    pTest.eval()
    let pTrainArr = pTrain.asArray(Float.self)
    let pTestArr = pTest.asArray(Float.self)

    // 4) Gitter ueber die Ebene, zurueck nach 30D (Rest 0), durch das echte Modell.
    var E: Float = 0
    for v in pTrainArr { E = max(E, abs(v)) }
    for v in pTestArr { E = max(E, abs(v)) }
    E *= (1 + margin)

    let res = resolution
    var xs: [Float] = [], ys: [Float] = []
    for i in 0..<res { xs.append(-E + 2 * E * Float(i) / Float(res - 1)) }
    for i in 0..<res { ys.append(-E + 2 * E * Float(i) / Float(res - 1)) }
    var pts: [Float] = []
    pts.reserveCapacity(res * res * 2)
    for iy in 0..<res {
        for ix in 0..<res {
            pts.append(xs[ix]); pts.append(ys[iy])
        }
    }
    let grid2d = MLXArray(pts).reshaped([res * res, 2])
    let grid30 = matmul(grid2d, V.T)          // G²×30, unsichtbare Koordinaten 0
    let f = svm.decisionFunction(sampler.transform(grid30))
    f.eval()
    let fArr = f.asArray(Float.self)

    // 12 Werte pro Zeile, wie in WriteVTK
    func vtkBlock(_ v: [Float]) -> [String] {
        var lines: [String] = []
        for start in stride(from: 0, to: v.count, by: 12) {
            let end = min(start + 12, v.count)
            lines.append(v[start..<end].map { String(format: "%.6g", $0) }
                .joined(separator: " "))
        }
        return lines
    }

    // 5) Flaeche: STRUCTURED_GRID, z-Koordinate = f(x)·height
    var out: [String] = []
    out.append("# vtk DataFile Version 3.0")
    out.append("WDBC RFF Entscheidungsflaeche in der PCA-Ebene, gamma=" + String(format: "%.3g", gamma))
    out.append("ASCII")
    out.append("DATASET STRUCTURED_GRID")
    out.append(String(format: "DIMENSIONS %d %d 1", res, res))
    out.append(String(format: "POINTS %d float", res * res))
    for iy in 0..<res {
        for ix in 0..<res {
            let i = iy * res + ix
            out.append(String(format: "%.6g %.6g %.6g", xs[ix], ys[iy], fArr[i] * height))
        }
    }
    out.append(String(format: "POINT_DATA %d", res * res))
    out.append("SCALARS f float 1")
    out.append("LOOKUP_TABLE default")
    out.append(contentsOf: vtkBlock(fArr))

    do {
        try out.joined(separator: "\n").write(toFile: surfacePath, atomically: true, encoding: .utf8)
        print("Flaeche geschrieben: \(surfacePath) (STRUCTURED_GRID \(res)x\(res), Hoehe = f·\(height))")
    } catch {
        print("Fehler beim Schreiben: \(error)")
    }

    // 6) Punkte: Trainings- UND Testpunkte auf Entscheidungshoehe
    let labelTrainArr = yTrain.asArray(Float.self)
    let labelTestArr = yTest.asArray(Float.self)
    let fTrainArr = decTrain.asArray(Float.self)
    let fTestArr = decTest.asArray(Float.self)
    let n = nTrain + nTest

    var labels: [Float] = [], decisions: [Float] = [], splits: [Float] = []
    labels.reserveCapacity(n); decisions.reserveCapacity(n); splits.reserveCapacity(n)

    var out2: [String] = []
    out2.append("# vtk DataFile Version 3.0")
    out2.append("WDBC Punkte auf der PCA-Entscheidungsflaeche, gamma=" + String(format: "%.3g", gamma))
    out2.append("ASCII")
    out2.append("DATASET POLYDATA")
    out2.append(String(format: "POINTS %d float", n))
    for i in 0..<nTrain {
        out2.append(String(format: "%.6g %.6g %.6g", pTrainArr[i * 2], pTrainArr[i * 2 + 1], fTrainArr[i] * height))
        labels.append(labelTrainArr[i])
        decisions.append(fTrainArr[i])
        splits.append(1)
    }
    for i in 0..<nTest {
        out2.append(String(format: "%.6g %.6g %.6g", pTestArr[i * 2], pTestArr[i * 2 + 1], fTestArr[i] * height))
        labels.append(labelTestArr[i])
        decisions.append(fTestArr[i])
        splits.append(0)
    }
    out2.append(String(format: "VERTICES %d %d", n, 2 * n))
    for i in 0..<n { out2.append("1 " + String(i)) }
    out2.append(String(format: "POINT_DATA %d", n))
    out2.append("SCALARS label float 1")
    out2.append("LOOKUP_TABLE default")
    out2.append(contentsOf: vtkBlock(labels))
    out2.append("SCALARS decision float 1")
    out2.append("LOOKUP_TABLE default")
    out2.append(contentsOf: vtkBlock(decisions))
    out2.append("SCALARS split float 1")
    out2.append("LOOKUP_TABLE default")
    out2.append(contentsOf: vtkBlock(splits))

    do {
        try out2.joined(separator: "\n").write(toFile: pointsPath, atomically: true, encoding: .utf8)
        print("Punkte geschrieben: \(pointsPath) (\(nTrain) Trainings- + \(nTest) Testpunkte, auf Hoehe ihrer Entscheidungswerte)")
        print("ParaView: Flaeche nach 'f' faerben (diverging), Contour f=0 = Grenze, Threshold -1.1 bis 1.1 = Support-Vektor-Schale; Punkte als Glyph, Color by 'label', nach 'split' filtern fuer Train/Test.")
    } catch {
        print("Fehler beim Schreiben: \(error)")
    }
}
