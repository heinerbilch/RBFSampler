
//
//  PlotComponentCurve.swift
//  RBFToolkit
//
//  Created by Heiner Strauss on 18.10.26.
//
import Foundation
import MLX

// MARK: - RBFSampler-Visualisierung: Fehler über m (Plotly-HTML)

/// Wie gut die Näherung je Anzahl Komponenten m wird: für dieselben
/// Punktpaare den mittleren |Kernel-Fehler| über m plotten – zusammen mit
/// der theoretischen Monte-Carlo-Referenz c/sqrt(m). Sichtbar: der Fehler
/// fällt mit 1/sqrt(m), d.h. Genauigkeit kostet quadratisch – 4x so viele
/// Komponenten halbieren den Fehler. Die Paare sind für alle m identisch
/// (gleicher SeededRNG), die Kurve ist also fair vergleichbar.
public func plotComponentCurve(x: MLXArray, gamma: Float,
                               ms: [Int] = [50, 100, 250, 500, 1000, 2000],
                               nPairs: Int = 300, seed: UInt64 = 11,
                               to path: String = "component_curve.html") {
    let n = x.shape[0], d = x.shape[1]
    let xArr = x.asArray(Float.self)

    // Punktpaare einmal ziehen – für alle m dieselben (faire Kurve)
    var rng = SeededRNG(seed: seed)
    var pairs: [(Int, Int)] = []
    pairs.reserveCapacity(nPairs)
    for _ in 0..<nPairs {
        let i = Int(rng.next() % UInt64(n))
        let j = Int(rng.next() % UInt64(n))
        pairs.append((i, j))
    }

    // Exakte Kernelwerte der Paare (Referenz, unabhängig von m)
    var kern: [Float] = []
    kern.reserveCapacity(nPairs)
    for (i, j) in pairs {
        var dist2: Float = 0
        for t in 0..<d {
            let dv = xArr[i * d + t] - xArr[j * d + t]
            dist2 += dv * dv
        }
        kern.append(exp(-gamma * dist2))
    }

    var errStr: [String] = [], refStr: [String] = []
    var firstErr: Float = 0
    for (g, m) in ms.enumerated() {
        let sampler = RBFSamplerMLX(dimension: d, nComponents: m,
                                    gamma: gamma, seed: 42)
        let z = sampler.transform(x)
        z.eval()
        let zArr = z.asArray(Float.self)
        var sumErr: Float = 0
        for p in 0..<nPairs {
            let (i, j) = pairs[p]
            var dot: Float = 0
            for k in 0..<m {
                dot += zArr[i * m + k] * zArr[j * m + k]
            }
            sumErr += abs(kern[p] - dot)
        }
        let err = sumErr / Float(nPairs)
        print(String(format: "m=%d: mittlerer Kernel-Fehler %.4f", m, err))
        if g == 0 { firstErr = err }
        errStr.append(String(format: "%.4g", err))
        refStr.append(String(format: "%.4g", firstErr * pow(Float(ms[0]) / Float(m), 0.5)))
    }

    let mStr = ms.map { String($0) }
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
      { type: 'scatter', mode: 'lines+markers', name: 'gemessener Fehler',
        x: [\(mStr.joined(separator: ", "))], y: [\(errStr.joined(separator: ", "))],
        marker: { size: 7, color: '#1f77b4' }, line: { width: 2, color: '#1f77b4' } },
      { type: 'scatter', mode: 'lines', name: 'Referenz c/sqrt(m)',
        x: [\(mStr.joined(separator: ", "))], y: [\(refStr.joined(separator: ", "))],
        line: { width: 1.5, color: '#d62728', dash: 'dash' } }
    ], {
      margin: { l: 60, r: 20, t: 60, b: 60 },
      title: "Monte-Carlo-Fehler über m (gamma=\(gamma), \(nPairs) Paare)",
      xaxis: { type: 'log', title: "Anzahl Komponenten m" },
      yaxis: { title: "mittlerer |Kernel-Fehler|" }
    }, { responsive: true });
      </script>
    </body>
    </html>
    """
    do {
        try html.write(toFile: path, atomically: true, encoding: .utf8)
        print("Fehlerkurve geschrieben: \(path) (im Browser öffnen)")
    } catch {
        print("Fehler beim Schreiben: \(error)")
    }
}
