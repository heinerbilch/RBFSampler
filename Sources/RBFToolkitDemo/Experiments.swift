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
protocol Experiment {
    static var name: String { get }
    static var about: String { get }
    init()
    func run() throws
}

enum Experiments {
    static let all: [any Experiment.Type] = [
        Pipeline.self,
        Visuals.self
    ]

    static func main() {
        let args = CommandLine.arguments
        guard args.count >= 2,
              let type = all.first(where: { $0.name == args[1].lowercased() }) else {
            print("Verfügbare Experimente:")
            for t in all {
                print("  swift run RBFToolkitDemo " + t.name + "  –  " + t.about)
            }
            exit(1)
        }
        print("Experiment: " + type.name)
        do {
            try type.init().run()
        } catch {
            print("Fehler: \(error)")
            exit(1)
        }
    }
}
