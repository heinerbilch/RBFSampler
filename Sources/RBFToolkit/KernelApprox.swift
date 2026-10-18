//
//  KernelApprox.swift
//  RBFToolkit
//
//  Created by Heiner Strauss on 18.10.26.
//
import Foundation
import MLX

// MARK: - RBFSampler-Visualisierung: Kernel-Näherung (Plotly-HTML)

/// Der Beweis, dass der Sampler das leistet, was er verspricht:
/// für zufällige Punktpaare den exakten Gauß-Kernel e^{-gamma||x-x'||^2}
/// gegen das Skalarprodukt der Feature-Vektoren plotten. Punkte auf der
/// Diagonalen = die Zufalls-Features rekonstruieren den Kernel. Der Fehler
/// ist Monte-Carlo und fällt mit 1/sqrt(m) – nComponents verdoppeln und
/// vergleichen. Die Konsole druckt zusätzlich max./mittleren Fehler.
public func plotKernelApprox(x: MLXArray, gamma: Float, nComponents: Int = 500,
                             nPairs: Int = 1500, seed: UInt64 = 11,
                             to path: String = "kernel_approx.html") {
    let n = x.shape[0], d = x.shape[1]
    let sampler = RBFSamplerMLX(dimension: d, nComponents: nComponents,
                                gamma: gamma, seed: 42)
    let z = sampler.transform(x)
    z.eval()
    let xArr = x.asArray(Float.self)
    let zArr = z.asArray(Float.self)

    var rng = SeededRNG(seed: seed)
    var ex: [String] = [], ap: [String] = []
    var maxErr: Float = 0, sumErr: Float = 0
    for _ in 0..<nPairs {
        let i = Int(rng.next() % UInt64(n))
        let j = Int(rng.next() % UInt64(n))
        var dist2: Float = 0
        for t in 0..<d {
            let dv = xArr[i * d + t] - xArr[j * d + t]
            dist2 += dv * dv
        }
        var dot: Float = 0
        for k in 0..<nComponents {
            dot += zArr[i * nComponents + k] * zArr[j * nComponents + k]
        }
        let kernel = exp(-gamma * dist2)
        maxErr = max(maxErr, abs(kernel - dot))
        sumErr += abs(kernel - dot)
        ex.append(String(format: "%.4g", kernel))
        ap.append(String(format: "%.4g", dot))
    }
    print(String(format: "Kernel-Näherung (m=%d): max. Fehler %.4f, mittlerer Fehler %.4f",
                 nComponents, maxErr, sumErr / Float(nPairs)))

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
      { type: 'scatter', mode: 'markers', name: 'Punktpaare',
        x: [\(ex.joined(separator: ", "))], y: [\(ap.joined(separator: ", "))],
        marker: { size: 3, color: '#1f77b4', opacity: 0.4 } },
      { type: 'scatter', mode: 'lines', name: 'exakt',
        x: [0, 1], y: [0, 1], line: { width: 1, color: '#d62728' } }
    ], {
      margin: { l: 60, r: 20, t: 60, b: 60 },
      title: "Zufalls-Features vs. exakter Gauß-Kernel (gamma=\(gamma), m=\(nComponents))",
      xaxis: { title: "exakter Kernel e^{-gamma||x-x'||^2}" },
      yaxis: { title: "z(x)·z(x')" }
    }, { responsive: true });
      </script>
    </body>
    </html>
    """
    do {
        try html.write(toFile: path, atomically: true, encoding: .utf8)
        print("Kernel-Vergleich geschrieben: \(path)")
    } catch {
        print("Fehler beim Schreiben: \(error)")
    }
}
