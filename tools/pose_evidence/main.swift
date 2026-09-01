import CoreGraphics
import CoreImage
import Foundation
import Vision

enum PoseEvidenceError: LocalizedError {
    case usage
    case unreadableImage(String)
    case cannotEncode

    var errorDescription: String? {
        switch self {
        case .usage:
            return "Usage: batch-color-pose-evidence <input>"
        case .unreadableImage(let path):
            return "Cannot read image: \(path)"
        case .cannotEncode:
            return "Cannot encode pose evidence"
        }
    }
}

@main
struct PoseEvidenceTool {
    static func normalizedImage(at url: URL, context: CIContext) throws -> CGImage {
        guard let source = CIImage(contentsOf: url, options: [.applyOrientationProperty: true]) else {
            throw PoseEvidenceError.unreadableImage(url.path)
        }
        let normalized = source.transformed(
            by: CGAffineTransform(translationX: -source.extent.minX, y: -source.extent.minY)
        )
        guard let image = context.createCGImage(normalized, from: normalized.extent) else {
            throw PoseEvidenceError.unreadableImage(url.path)
        }
        return image
    }

    static func pixelPoint(
        _ point: VNRecognizedPoint,
        width: Int,
        height: Int
    ) -> [String: Any] {
        [
            "x": Double(point.location.x) * Double(width),
            "y": (1.0 - Double(point.location.y)) * Double(height),
            "confidence": Double(point.confidence),
        ]
    }

    static func bodyPoint(
        _ observation: VNHumanBodyPoseObservation,
        _ joint: VNHumanBodyPoseObservation.JointName,
        width: Int,
        height: Int
    ) -> [String: Any]? {
        guard let point = try? observation.recognizedPoint(joint), point.confidence > 0 else {
            return nil
        }
        return pixelPoint(point, width: width, height: height)
    }

    static func handPoint(
        _ observation: VNHumanHandPoseObservation,
        _ joint: VNHumanHandPoseObservation.JointName,
        width: Int,
        height: Int
    ) -> [String: Any]? {
        guard let point = try? observation.recognizedPoint(joint), point.confidence > 0 else {
            return nil
        }
        return pixelPoint(point, width: width, height: height)
    }

    static func faceLandmarkPoints(
        _ region: VNFaceLandmarkRegion2D?,
        box: CGRect,
        width: Int,
        height: Int
    ) -> [[String: Any]] {
        guard let region else { return [] }
        return region.normalizedPoints.map { point in
            let normalizedX = Double(box.minX) + Double(point.x) * Double(box.width)
            let normalizedY = Double(box.minY) + Double(point.y) * Double(box.height)
            return [
                "x": normalizedX * Double(width),
                "y": (1.0 - normalizedY) * Double(height),
            ]
        }
    }

    static func run() throws {
        let arguments = CommandLine.arguments
        guard arguments.count == 2 else { throw PoseEvidenceError.usage }
        let inputURL = URL(fileURLWithPath: arguments[1])
        let ciContext = CIContext(options: [.useSoftwareRenderer: false])
        let source = try normalizedImage(at: inputURL, context: ciContext)

        let faceRequest = VNDetectFaceLandmarksRequest()
        let bodyRequest = VNDetectHumanBodyPoseRequest()
        let handRequest = VNDetectHumanHandPoseRequest()
        handRequest.maximumHandCount = 2
        let handler = VNImageRequestHandler(cgImage: source, orientation: .up, options: [:])
        try handler.perform([faceRequest, bodyRequest, handRequest])

        let bodyJoints: [(String, VNHumanBodyPoseObservation.JointName)] = [
            ("neck", .neck),
            ("root", .root),
            ("leftShoulder", .leftShoulder),
            ("leftElbow", .leftElbow),
            ("leftWrist", .leftWrist),
            ("rightShoulder", .rightShoulder),
            ("rightElbow", .rightElbow),
            ("rightWrist", .rightWrist),
            ("leftHip", .leftHip),
            ("rightHip", .rightHip),
            ("leftKnee", .leftKnee),
            ("rightKnee", .rightKnee),
            ("leftAnkle", .leftAnkle),
            ("rightAnkle", .rightAnkle),
        ]
        let handJoints: [(String, VNHumanHandPoseObservation.JointName)] = [
            ("wrist", .wrist),
            ("thumbCMC", .thumbCMC), ("thumbMP", .thumbMP),
            ("thumbIP", .thumbIP), ("thumbTip", .thumbTip),
            ("indexMCP", .indexMCP), ("indexPIP", .indexPIP),
            ("indexDIP", .indexDIP), ("indexTip", .indexTip),
            ("middleMCP", .middleMCP), ("middlePIP", .middlePIP),
            ("middleDIP", .middleDIP), ("middleTip", .middleTip),
            ("ringMCP", .ringMCP), ("ringPIP", .ringPIP),
            ("ringDIP", .ringDIP), ("ringTip", .ringTip),
            ("littleMCP", .littleMCP), ("littlePIP", .littlePIP),
            ("littleDIP", .littleDIP), ("littleTip", .littleTip),
        ]

        let bodies: [[String: Any]] = (bodyRequest.results ?? []).map { observation in
            var points: [String: Any] = [:]
            for (name, joint) in bodyJoints {
                if let point = bodyPoint(observation, joint, width: source.width, height: source.height) {
                    points[name] = point
                }
            }
            return ["confidence": Double(observation.confidence), "points": points]
        }

        let hands: [[String: Any]] = (handRequest.results ?? []).map { observation in
            var points: [String: Any] = [:]
            for (name, joint) in handJoints {
                if let point = handPoint(observation, joint, width: source.width, height: source.height) {
                    points[name] = point
                }
            }
            return ["confidence": Double(observation.confidence), "points": points]
        }

        let faces: [[String: Any]] = (faceRequest.results ?? []).map { observation in
            let box = observation.boundingBox
            let landmarks = observation.landmarks
            return [
                "confidence": Double(observation.confidence),
                "bbox": [
                    "x": Double(box.minX) * Double(source.width),
                    "y": (1.0 - Double(box.maxY)) * Double(source.height),
                    "width": Double(box.width) * Double(source.width),
                    "height": Double(box.height) * Double(source.height),
                ],
                "landmarks": [
                    "faceContour": faceLandmarkPoints(landmarks?.faceContour, box: box, width: source.width, height: source.height),
                    "leftEye": faceLandmarkPoints(landmarks?.leftEye, box: box, width: source.width, height: source.height),
                    "rightEye": faceLandmarkPoints(landmarks?.rightEye, box: box, width: source.width, height: source.height),
                    "leftEyebrow": faceLandmarkPoints(landmarks?.leftEyebrow, box: box, width: source.width, height: source.height),
                    "rightEyebrow": faceLandmarkPoints(landmarks?.rightEyebrow, box: box, width: source.width, height: source.height),
                    "nose": faceLandmarkPoints(landmarks?.nose, box: box, width: source.width, height: source.height),
                    "outerLips": faceLandmarkPoints(landmarks?.outerLips, box: box, width: source.width, height: source.height),
                    "innerLips": faceLandmarkPoints(landmarks?.innerLips, box: box, width: source.width, height: source.height),
                ],
            ]
        }

        let payload: [String: Any] = [
            "schema": 1,
            "backend": "apple-vision-pose-evidence-v1",
            "width": source.width,
            "height": source.height,
            "faces": faces,
            "bodies": bodies,
            "hands": hands,
        ]
        guard JSONSerialization.isValidJSONObject(payload) else { throw PoseEvidenceError.cannotEncode }
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
        guard let output = String(data: data, encoding: .utf8) else { throw PoseEvidenceError.cannotEncode }
        print(output)
    }

    static func main() {
        do {
            try run()
        } catch {
            FileHandle.standardError.write(Data("pose-evidence: \(error.localizedDescription)\n".utf8))
            Foundation.exit(1)
        }
    }
}
