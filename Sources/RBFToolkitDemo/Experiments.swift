//
//  Experiments.swift
//  RBFToolkitDemo
//
//  Created by Heiner Strauss on 18.10.26.
//
import Foundation

// MARK: - Experiment-Registry

/// Jedes Experiment ist eine eigene Datei und folgt diesem Protokoll.
/// Neue Experimente nur hier in all: registrieren – main.swift bleibt unberührt.
/// run(steps:) erhält alle CLI-Argumente hinter dem Experimentnamen;
/// eine leere Liste bedeutet "alles ausführen".
protocol Experiment {
    static var name: String { get }
    static var about: String { get }
    init()
    func run(steps: [String]) throws
}

enum Experiments {
    static let all: [any Experiment.Type] = [
        Pipeline.self,
        Visuals.self,
        Cancer.self
    ]

    static func main() {
        let args = CommandLine.arguments
        guard args.count >= 2,
              let type = all.first(where: { $0.name == args[1].lowercased() }) else {
            print("Verfügbare Experimente (Schritte optional):")
            for t in all {
                print("  swift run RBFToolkitDemo " + t.name + " [schritt …]  –  " + t.about)
            }
            exit(1)
        }
        let steps = args.count > 2 ? Array(args.dropFirst(2)) : []
        print("Experiment: " + type.name + (steps.isEmpty ? "" : " (" + steps.joined(separator: ", ") + ")"))
        do {
            try type.init().run(steps: steps)
        } catch {
            print("Fehler: \(error)")
            exit(1)
        }
    }
}
