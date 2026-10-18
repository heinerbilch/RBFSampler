//
//  RbfWaves.swift
//  RBFToolkit
//
//  Created by Heiner Strauss on 18.10.26.
//
import Foundation
import MLX

// MARK: - RBFSampler-Visualisierung: Wellen in 1D (Plotly-HTML)

/// Zeigt die ersten Komponenten eines 1D-Samplers als Kurven über x.
/// Macht sichtbar, was gamma physikalisch bedeutet: kleine gamma = lange,
/// glatte Wellen (globale Geometrie), große gamma = schnelle Schwingung
/// (Rauschen). Die Mittelung über alle Komponenten ergibt den Kernel.
public func plotRbfWaves(gamma: Float, nComponents: Int = 500, show: Int = 6,
                         to path: String = "waves.html", extent: Float = 4) {
    let n = 400
    var xs: [Float] = []
    for i in 0..<n { xs.append(-extent + 2 * extent * Float(i) / Float(n - 1)) }
    let x = MLXArray(xs).reshaped([n, 1])
    let sampler = RBFSamplerMLX(dimension: 1, nComponents: nComponents,
                                gamma: gamma, seed: 42)
    let z = sampler.transform(x)
    z.eval()
    let zArr = z.asArray(Float.self)

    let colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    var traces: [String] = []
    for k in 0..<min(show, nComponents) {
        let xStr = xs.map { String(format: "%.4g", $0) }
        let yStr = (0..<n).map { String(format: "%.4g", zArr[$0 * nComponents + k]) }
        traces.append("  { type: 'scatter', mode: 'lines', name: 'z_" + String(k) + "',\n" +
            "    x: [" + xStr.joined(separator: ", ") + "],\n" +
            "    y: [" + yStr.joined(separator: ", ") + "],\n" +
            "    line: { width: 1.5, color: '" + colors[k % colors.count] + "' } }")
    }

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
    \(traces.joined(separator: ",\n"))
    ], {
      margin: { l: 50, r: 20, t: 60, b: 50 },
      title: "RBFSampler-Wellen (gamma=\(gamma), \(nComponents) Komponenten)",
      xaxis: { title: "x" }, yaxis: { title: "z_k(x)" }
    }, { responsive: true });
      </script>
    </body>
    </html>
    """
    do {
        try html.write(toFile: path, atomically: true, encoding: .utf8)
        print("Wellen-Plot geschrieben: \(path)")
    } catch {
        print("Fehler beim Schreiben: \(error)")
    }
}
