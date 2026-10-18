
//
//  PlotDecisionLandscape.swift
//  RBFToolkit
//
//  Created by Heiner Strauss on 18.10.26.
//
import Foundation
import MLX

// MARK: - RBFSampler-Visualisierung: Entscheidungslandschaft (Plotly-HTML)

/// Das ganze Verfahren in einem Bild, für d=2: Blobs -> zufällige Features ->
/// lineare SVM im Feature-Raum -> Entscheidungsfunktion f(x) über dem
/// Eingabegitter als Farblandschaft. Sichtbar wird die Kernaussage von RFF:
/// im Feature-Raum ist die Grenze eine Ebene, zurück im Eingaberaum ist sie
/// eine gewellte Kurve – gebaut aus denselben Kosinuswellen wie im Wellen-Plot.
/// Durchgezogene Linie: f=0 (Grenze). Gepunktet: |f|=1 (Marge) – erscheint nur,
/// wenn die Entscheidungswerte dorthin kommen (RFF-Features liegen auf der
/// Einheitssphäre, die Beträge sind gedeckelt). Kleines gamma = glatte Grenze,
/// großes gamma = gezackte Grenze. Ein Aufruf pro gamma-Wert.
public func plotDecisionLandscape(gamma: Float, shift: Float = 1.2, nTrain: Int = 4000,
                                  nComponents: Int = 500, epochs: Int = 20,
                                  extent: Float = 4, resolution: Int = 100,
                                  to path: String = "boundary.html") {
    // 1) Trainingsdaten in 2D: so viel Überlappung, dass die Grenze interessiert
    let (x, y) = makeBlobs(n: nTrain, d: 2, shift: shift)
    let n = x.shape[0]

    // 2) Features + lineares SVM-Training – exakt die Pipeline aus dem Paket
    let sampler = RBFSamplerMLX(dimension: 2, nComponents: nComponents,
                                gamma: gamma, seed: 42)
    let z = sampler.transform(x)
    z.eval()
    var svm = LinearSGDSVM(featureCount: nComponents, lambda: 1e-4, learningRate: 0.05)
    svm.fit(z, y, epochs: epochs, batchSize: 256)
    let pred = svm.predict(z)
    pred.eval()
    let acc = 100.0 * Float((pred * y .> 0).sum().item(Int.self)) / Float(n)
    print(String(format: "Landschaft gamma=%.3g: Trainingsgenauigkeit %.1f%%", gamma, acc))

    // 3) Gitter durch dieselbe Feature-Map und decisionFunction schieben
    let res = resolution
    var xs: [Float] = [], ys: [Float] = []
    for i in 0..<res { xs.append(-extent + 2 * extent * Float(i) / Float(res - 1)) }
    for i in 0..<res { ys.append(-extent + 2 * extent * Float(i) / Float(res - 1)) }
    var pts: [Float] = []
    pts.reserveCapacity(res * res * 2)
    for iy in 0..<res {
        for ix in 0..<res {
            pts.append(xs[ix]); pts.append(ys[iy])
        }
    }
    let xg = MLXArray(pts).reshaped([res * res, 2])
    let f = svm.decisionFunction(sampler.transform(xg))
    f.eval()
    let fArr = f.asArray(Float.self)
    let fMin = f.min().item(Float.self)
    let fMax = f.max().item(Float.self)
    let marge = (fMax >= 1 || fMin <= -1) ? "sichtbar" : "ausserhalb des Wertebereichs (Beträge gedeckelt)"
    print(String(format: "f über dem Gitter: min=%.2f, max=%.2f – Marge |f|=1: %@", fMin, fMax, marge))

    // 4) z-Matrix für Plotly: Zeile = y, Spalte = x
    let xStr = xs.map { String(format: "%.4g", $0) }
    let yStr = ys.map { String(format: "%.4g", $0) }
    var zRows: [String] = []
    for iy in 0..<res {
        let row = (0..<res).map { String(format: "%.4g", fArr[iy * res + $0]) }
        zRows.append("    [" + row.joined(separator: ", ") + "]")
    }

    // 5) Punkte nach Klasse getrennt (wie plotScatter)
    let xArr = x.asArray(Float.self)
    let yArr = y.asArray(Float.self)
    var pX: [String] = [], pY: [String] = [], nX: [String] = [], nY: [String] = []
    for i in 0..<n {
        let a = String(format: "%.4g", xArr[i * 2])
        let b = String(format: "%.4g", xArr[i * 2 + 1])
        if yArr[i] > 0 { pX.append(a); pY.append(b) }
        else           { nX.append(a); nY.append(b) }
    }

    let contourSize = max((fMax - fMin) / 12, 0.05)
    let accStr = String(format: "%.1f", acc)
    let extStr = String(format: "%g", extent)
    let zBlock = zRows.joined(separator: ",\n")
    let html = """
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8">
      <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    </head>
    <body style="margin:0">
      <div id="plot" style="width:100vw;height:100vh"></div>
      <script>
    Plotly.newPlot('plot', [
      { type: 'contour', name: 'f(x)', showscale: true, showlegend: false,
        x: [\(xStr.joined(separator: ", "))], y: [\(yStr.joined(separator: ", "))],
        z: [
    \(zBlock)
        ],
        colorscale: [[0, '#67001f'], [0.25, '#d6604d'], [0.5, '#f7f7f7'], [0.75, '#4393c3'], [1, '#053061']],
        contours: { start: \(String(format: "%.4g", fMin)), end: \(String(format: "%.4g", fMax)), size: \(String(format: "%.4g", contourSize)) },
        colorbar: { title: 'f(x)' } },
      { type: 'contour', name: 'Grenze f=0', showscale: false, showlegend: true,
        x: [\(xStr.joined(separator: ", "))], y: [\(yStr.joined(separator: ", "))],
        z: [
    \(zBlock)
        ],
        contours: { start: -2, end: 2, size: 2, coloring: 'lines' },
        line: { color: '#222222', width: 2.5 } },
      { type: 'contour', name: 'Marge |f|=1', showscale: false, showlegend: true,
        x: [\(xStr.joined(separator: ", "))], y: [\(yStr.joined(separator: ", "))],
        z: [
    \(zBlock)
        ],
        contours: { start: -1, end: 1, size: 2, coloring: 'lines' },
        line: { color: '#555555', width: 1.5, dash: 'dot' } },
      { type: 'scatter', mode: 'markers', name: 'Klasse +1',
        x: [\(pX.joined(separator: ", "))], y: [\(pY.joined(separator: ", "))],
        marker: { size: 3.5, color: '#1f77b4', opacity: 0.6 } },
      { type: 'scatter', mode: 'markers', name: 'Klasse -1',
        x: [\(nX.joined(separator: ", "))], y: [\(nY.joined(separator: ", "))],
        marker: { size: 3.5, color: '#d62728', opacity: 0.6 } }
    ], {
      margin: { l: 50, r: 20, t: 60, b: 50 },
      title: "RFF-Entscheidungslandschaft (gamma=\(gamma), m=\(nComponents), Genauigkeit \(accStr)%)",
      xaxis: { title: "x[0]", range: [-\(extStr), \(extStr)] },
      yaxis: { title: "x[1]", range: [-\(extStr), \(extStr)], scaleanchor: "x", scaleratio: 1 },
      legend: { orientation: "h" }
    }, { responsive: true });
      </script>
    </body>
    </html>
    """
    do {
        try html.write(toFile: path, atomically: true, encoding: .utf8)
        print("Landschaft geschrieben: \(path) (im Browser öffnen)")
    } catch {
        print("Fehler beim Schreiben: \(error)")
    }
}
