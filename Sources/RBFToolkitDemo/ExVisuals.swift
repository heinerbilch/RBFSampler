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

/// Die RBFSampler-Erklärstücke, einzeln oder komplett. Ohne Schritt-Argument
/// läuft alles; mit Schritten nur die benannten – zum schnellen Iterieren,
/// ohne jedes Mal alle Ausgaben neu zu erzeugen.
struct Visuals: Experiment {
    static let name = "visuals"
    static let about = "Visualisierungen: waves field kernel omega boundary surface epochs components (ohne Argument: alle)"

    init() {}

    private static let steps: [(name: String, about: String)] = [
        ("waves", "1D-Wellen: gamma klein vs. gross"),
        ("field", "2D-Wellenfeld (VTK) für ParaView"),
        ("kernel", "Kernel-Näherung gegen den exakten Gauß-Kernel"),
        ("omega", "Frequenzraum: gezogene Wellenvektoren"),
        ("boundary", "Entscheidungslandschaft: Grenze + Marge"),
        ("surface", "3D-Entscheidungsflaeche (VTK) für ParaView"),
        ("epochs", "Trainingsepochen als VTK-Zeitsequenz (ParaView-Animation)"),
        ("components", "Fehlerkurve über m mit 1/sqrt(m)-Referenz")
    ]

    func run(steps: [String]) throws {
        let toRun = steps.isEmpty ? Visuals.steps.map { $0.name } : steps
        for s in toRun { try runStep(s) }
    }

    private func runStep(_ step: String) throws {
        switch step {
        case "waves":
            plotRbfWaves(gamma: 0.05, to: desktopPath("waves_small_gamma.html"))
            plotRbfWaves(gamma: 2.0, to: desktopPath("waves_large_gamma.html"))
        case "field":
            writeVtkWaveField(gamma: 0.25, to: desktopPath("wave.vtk"))
        case "kernel":
            let (x, _) = makeBlobs(n: 20_000, d: 32)
            plotKernelApprox(x: x, gamma: 0.003, to: desktopPath("kernel_approx.html"))
        case "omega":
            plotOmegaCloud(to: desktopPath("omega_cloud.html"))
        case "boundary":
            plotDecisionLandscape(gamma: 0.05, to: desktopPath("boundary_small_gamma.html"))
            plotDecisionLandscape(gamma: 2.0, to: desktopPath("boundary_large_gamma.html"))
        case "surface":
            writeVtkDecisionSurface(gamma: 0.05,
                                    toSurface: desktopPath("surface_small_gamma.vtk"),
                                    toPoints: desktopPath("surface_points_small_gamma.vtk"))
            writeVtkDecisionSurface(gamma: 2.0,
                                    toSurface: desktopPath("surface_large_gamma.vtk"),
                                    toPoints: desktopPath("surface_points_large_gamma.vtk"))
        case "epochs":
            writeVtkEpochAnimation(gamma: 0.05,
                                   toDirectory: desktopPath("epoch_anim_small_gamma"))
            writeVtkEpochAnimation(gamma: 2.0,
                                   toDirectory: desktopPath("epoch_anim_large_gamma"))
        case "components":
            let (x, _) = makeBlobs(n: 20_000, d: 32)
            plotComponentCurve(x: x, gamma: 0.003, to: desktopPath("component_curve.html"))
        default:
            print("Unbekannter Schritt: " + step + ". Verfügbare Schritte:")
            for s in Visuals.steps { print("  " + s.name + "  –  " + s.about) }
        }
    }
}
