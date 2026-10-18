//
//  ExVisuals.swift
//  RBFToolkitDemo
//
//  Created by Heiner Strauss on 18.10.26.
//
import Foundation
import MLX
import RBFToolkit

// MARK: - Experiment: Visuals

/// Die RBFSampler-Erklärstücke: Wellen für zwei gamma-Werte, Wellenfeld
/// für ParaView, und die Kernel-Näherung gegen den exakten Gauß-Kernel.
struct Visuals: Experiment {
    static let name = "visuals"
    static let about = "RBFSampler-Visualisierungen: Wellen, Wellenfeld, Kernel-Näherung"

    init() {}

    func run() throws {
        // 1) Wellen: gamma-Effekt direkt sichtbar (langsam vs. schnell)
        plotRbfWaves(gamma: 0.05, to: desktopPath("waves_small_gamma.html"))
        plotRbfWaves(gamma: 2.0, to: desktopPath("waves_large_gamma.html"))

        // 2) Eine Komponente als 2D-Wellenfeld für ParaView
        writeVtkWaveField(gamma: 0.25, to: desktopPath("wave.vtk"))

        // 3) Kernel-Näherung auf den Demo-Blobs; gamma=0.003 ist der Wert,
        //    den das Pipeline-Experiment per CV-Tuning findet.
        let (x, _) = makeBlobs(n: 20_000, d: 32)
        plotKernelApprox(x: x, gamma: 0.003, to: desktopPath("kernel_approx.html"))
    }
}
