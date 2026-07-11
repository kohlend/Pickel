//
//  ScanBenchmark.swift
//  Pickex
//
//  On-device performance measurement for the library scan (Step-C §6).
//  The conversion/CI environment is Linux — Photos/Vision/Core ML timings can
//  only be measured on a real iPhone. Drop this into the app, run
//  `ScanBenchmark.run(...)` once (e.g. behind a debug button), and read the
//  numbers from the console / returned report.
//
//  Measures:
//  • wall time total and per-1000-photos rate
//  • cache hit ratio (run twice: 2nd run shows the cached-scan speed)
//  • peak resident memory (via task_vm_info.phys_footprint)
//  • how many assets were iCloud-only
//
//  Background note: without a BGProcessingTask, iOS suspends the app shortly
//  after it leaves the foreground and the scan pauses until reopened (the
//  cache makes resume cheap). Whether v1 ships "keep the app open" or a
//  BGProcessingTask is a product decision — see README.
//

import Foundation
import Photos

public struct ScanBenchmarkReport: CustomStringConvertible {
    public let totalAssets: Int
    public let processed: Int
    public let matches: Int
    public let skippedNotLocal: Int
    public let servedFromCache: Int
    public let wallSeconds: Double
    public let peakMemoryMB: Double

    public var secondsPer1000: Double {
        processed > 0 ? wallSeconds / Double(processed) * 1000 : 0
    }

    public var description: String {
        """
        ── Pickex scan benchmark ─────────────────────────
        assets            \(totalAssets)  (processed \(processed))
        matches           \(matches)
        iCloud-only skips \(skippedNotLocal)
        cache hits        \(servedFromCache)
        wall time         \(String(format: "%.1f", wallSeconds)) s \
        (\(String(format: "%.1f", secondsPer1000)) s / 1000 photos)
        peak memory       \(String(format: "%.0f", peakMemoryMB)) MB
        ──────────────────────────────────────────────────
        """
    }
}

public enum ScanBenchmark {

    /// Runs a full scan and prints/returns the report. Call once with a fresh
    /// cache ("cold"), then again ("warm") to measure the cached path.
    @discardableResult
    public static func run(scanner: LibraryScanner,
                           profile: ReferenceProfile) async throws -> ScanBenchmarkReport {
        let start = Date()
        var last: ScanProgress?
        var matches = 0
        var peak = currentFootprintMB()

        for try await event in scanner.scanEvents(against: profile) {
            switch event {
            case .progress(let p):
                last = p
                peak = max(peak, currentFootprintMB())
            case .match:
                matches += 1
            }
        }

        let report = ScanBenchmarkReport(
            totalAssets: last?.total ?? 0,
            processed: last?.processed ?? 0,
            matches: matches,
            skippedNotLocal: last?.skippedNotLocal ?? 0,
            servedFromCache: last?.servedFromCache ?? 0,
            wallSeconds: Date().timeIntervalSince(start),
            peakMemoryMB: peak)
        print(report)
        return report
    }

    /// Current physical footprint in MB (what Xcode's memory gauge shows).
    public static func currentFootprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = TASK_VM_INFO_COUNT
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / 1_048_576
    }
}

private let TASK_VM_INFO_COUNT = mach_msg_type_number_t(
    MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
