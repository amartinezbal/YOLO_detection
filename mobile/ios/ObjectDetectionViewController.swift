// ObjectDetectionViewController.swift
// ─────────────────────────────────────────────────────────
// Native iOS: AVFoundation + Vision + CoreML
// Trained on Objects365 (365 classes)
//
// Setup:
//   1. Add yolov8n.mlpackage to your Xcode target
//   2. Add NSCameraUsageDescription to Info.plist
// ─────────────────────────────────────────────────────────

import UIKit
import AVFoundation
import Vision
import CoreML

// ─── Objects365 class names (365 classes) ──────────────────
private let objects365Classes: [String] = [
    "Person","Sneakers","Chair","Other Shoes","Hat","Car","Lamp","Glasses",
    "Bottle","Desk","Cup","Street Lights","Cabinet/shelf","Handbag/Satchel",
    "Bracelet","Plate","Picture/Frame","Helmet","Book","Gloves","Storage box",
    "Boat","Leather Shoes","Flower","Bench","Potted Plant","Bowl/Basin","Flag",
    "Pillow","Boots","Vase","Microphone","Necklace","Ring","SUV","Wine Glass",
    "Belt","Monitor/TV","Backpack","Umbrella","Traffic Light","Speaker","Watch",
    "Tie","Trash bin Can","Slippers","Bicycle","Stool","Barrel/bucket","Van",
    "Couch","Sandals","Basket","Drum","Pen/Pencil","Bus","Wild Bird","High Heels",
    "Motorcycle","Guitar","Carpet","Cell Phone","Bread","Camera","Canned",
    "Billiards","Candle","Stuffed Animal","Spoon","Hot dog","Calories Counter",
    "Donut","Stroller","Medical Equipment","Fire Extinguisher","Computer Box",
    "Earphone","Painting","Watermelon","Potato","Apple","Orange","Lemon",
    "Durian","Avocado","Strawberry","Banana","Pitcher","Mango","Baseball Glove",
    "Skateboard","Soccer","Globe","Chess","Arrow","Pliers","Saw","Soccer Shoe",
    "Jug","Stapler","Fork","Comb","Pot/Pan","Padlock","Broom","Refrigerator",
    "Oven/Stove","Washing Machine","Blender","Microwave","Laptop","Keyboard",
    "Mouse","Remote","Clock","Ruler","Scissors","Tape","Eraser","Pencil Case",
    "Backpack","Suitcase","Umbrella","Flashlight","Canteen","Compass",
    "Magnifying Glass","Telescope","Binoculars","Map","Trophy","Medal",
    "Certificate","Stamp","Envelope","Postcard","Newspaper","Magazine",
    "Notebook","Diary","Album","Wallet","Coin","Key","Lock","Safe",
    "Piggy Bank","ATM Card","Gift Box","Ribbon","Balloon","Party Hat",
    "Confetti","Fireworks","Sparkler","Lantern","Christmas Tree","Wreath",
    "Snowman","Jack-o-lantern","Easter Egg","Wedding Cake","Champagne",
    "Wine Bottle","Beer","Cocktail","Coffee","Tea","Juice","Smoothie",
    "Soda Can","Water Bottle","Thermos","Kettle","Teapot","Coffee Maker",
    "Blender","Toaster","Waffle Maker","Rice Cooker","Slow Cooker",
    "Pressure Cooker","Wok","Frying Pan","Saucepan","Baking Sheet",
    "Mixing Bowl","Colander","Cutting Board","Knife Block","Peeler",
    "Grater","Mortar and Pestle","Measuring Cup","Rolling Pin","Spatula",
    "Whisk","Ladle","Tongs","Oven Mitt","Apron","Dining Table","Chair",
    "Bar Stool","Rocking Chair","Armchair","Sofa","Loveseat","Chaise Lounge",
    "Ottoman","Coffee Table","End Table","Console Table","Bookshelf",
    "Filing Cabinet","Wardrobe","Dresser","Nightstand","Bed Frame",
    "Mattress","Pillow","Blanket","Comforter","Duvet","Sheet","Curtain",
    "Blind","Rug","Doormat","Floor Lamp","Desk Lamp","Ceiling Light",
    "Chandelier","Wall Sconce","Candle Holder","Mirror","Picture Frame",
    "Wall Clock","Coat Rack","Umbrella Stand","Shoe Rack","Laundry Basket",
    "Iron","Ironing Board","Washing Machine","Dryer","Vacuum Cleaner",
    "Mop","Broom","Dustpan","Bucket","Sponge","Cleaning Spray",
    "Trash Can","Recycling Bin","Paper Shredder","Printer","Scanner",
    "Fax Machine","Telephone","Answering Machine","Router","Modem",
    "Power Strip","Extension Cord","Smart Speaker","Streaming Device",
    "Game Console","VR Headset","Projector","Whiteboard","Corkboard",
    "Easel","Drawing Tablet","Stylus","Camera Tripod","Ring Light",
    "Green Screen","Podcast Mic","DSLR Camera","Action Camera","Drone",
    "Gimbal","Lens Filter","Camera Bag","Memory Card","External Drive",
    "NAS Drive","Server Rack","UPS","Oscilloscope","Multimeter",
    "Soldering Iron","Breadboard","Arduino","Raspberry Pi","3D Printer",
    "Laser Cutter","CNC Machine","Lathe","Drill Press","Band Saw",
    "Jigsaw","Circular Saw","Table Saw","Router Table","Workbench",
    "Vise","Clamp","Hammer","Screwdriver","Wrench","Pliers","Level",
    "Tape Measure","Square","Protractor","Caulk Gun","Paint Brush",
    "Paint Roller","Drop Cloth","Ladder","Scaffold","Wheelbarrow",
    "Garden Hose","Sprinkler","Watering Can","Shovel","Rake","Hoe",
    "Trowel","Pruners","Hedge Trimmer","Lawn Mower","Leaf Blower",
    "Chainsaw","Axe","Splitting Maul","Log Splitter","Wood Chipper",
]

private let boxColors: [UIColor] = [
    .systemRed, .systemOrange, .systemYellow, .systemGreen,
    .systemTeal, .systemCyan, .systemBlue, .systemIndigo, .systemPurple,
    .systemPink, .systemBrown, .systemMint,
]

// ─────────────────────────────────────────────────────────
final class ObjectDetectionViewController: UIViewController {

    private let captureSession  = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private let videoQueue      = DispatchQueue(label: "com.demo.yolo.video", qos: .userInteractive)
    private var detectionRequest: VNCoreMLRequest?
    private let overlayView     = UIView()
    private var boxLayers: [CAShapeLayer] = []
    private var labelLayers: [CATextLayer] = []
    private let hudLabel        = UILabel()
    private var frameCount      = 0
    private var lastFPSTime     = Date()

    // ─── Lifecycle ────────────────────────────────────────
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        setupCamera()
        setupOverlay()
        setupHUD()
        loadModel()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
        overlayView.frame  = view.bounds
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        videoQueue.async { self.captureSession.startRunning() }
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        captureSession.stopRunning()
    }

    // ─── Camera ───────────────────────────────────────────
    private func setupCamera() {
        captureSession.sessionPreset = .hd1280x720
        guard
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            let input  = try? AVCaptureDeviceInput(device: device)
        else { return }
        captureSession.addInput(input)
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)
        captureSession.addOutput(output)
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
    }

    private func setupOverlay() {
        overlayView.backgroundColor = .clear
        overlayView.isUserInteractionEnabled = false
        view.addSubview(overlayView)
    }

    private func setupHUD() {
        hudLabel.backgroundColor   = UIColor.black.withAlphaComponent(0.6)
        hudLabel.textColor         = .white
        hudLabel.font              = .monospacedSystemFont(ofSize: 13, weight: .semibold)
        hudLabel.textAlignment     = .center
        hudLabel.layer.cornerRadius = 14
        hudLabel.clipsToBounds     = true
        hudLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(hudLabel)
        NSLayoutConstraint.activate([
            hudLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -16),
            hudLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            hudLabel.heightAnchor.constraint(equalToConstant: 40),
            hudLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 240),
        ])
    }

    // ─── Load CoreML model ────────────────────────────────
    private func loadModel() {
        // "yolov8n" must match the .mlpackage filename in your Xcode target
        guard
            let modelURL   = Bundle.main.url(forResource: "yolov8n", withExtension: "mlpackage"),
            let coreMLModel = try? MLModel(contentsOf: modelURL),
            let vnModel    = try? VNCoreMLModel(for: coreMLModel)
        else {
            print("⚠️  CoreML model not found. Add yolov8n.mlpackage to your Xcode target.")
            return
        }
        let req = VNCoreMLRequest(model: vnModel) { [weak self] req, _ in
            self?.handleDetections(req.results)
        }
        req.imageCropAndScaleOption = .scaleFill
        detectionRequest = req
    }

    // ─── Process Vision results ───────────────────────────
    private func handleDetections(_ results: [VNObservation]?) {
        guard let obs = results as? [VNRecognizedObjectObservation] else { return }
        let filtered = obs.filter { $0.confidence > 0.35 }

        frameCount += 1
        let now     = Date()
        var fpsText = ""
        if now.timeIntervalSince(lastFPSTime) >= 1.0 {
            fpsText    = "\(frameCount) FPS"
            frameCount = 0
            lastFPSTime = now
        }

        DispatchQueue.main.async { [weak self] in
            self?.drawBoxes(filtered, fps: fpsText)
        }
    }

    // ─── Draw bounding boxes ──────────────────────────────
    private func drawBoxes(_ obs: [VNRecognizedObjectObservation], fps: String) {
        boxLayers.forEach   { $0.removeFromSuperlayer() }
        labelLayers.forEach { $0.removeFromSuperlayer() }
        boxLayers.removeAll()
        labelLayers.removeAll()

        let bounds = overlayView.bounds

        for o in obs {
            guard let top = o.labels.first else { continue }

            let rect    = VNImageRectForNormalizedRect(o.boundingBox, Int(bounds.width), Int(bounds.height))
            let flipped = CGRect(x: rect.minX, y: bounds.height - rect.maxY,
                                 width: rect.width, height: rect.height)

            // Match class index to color
            let classIdx = objects365Classes.firstIndex(of: top.identifier) ?? 0
            let color    = boxColors[classIdx % boxColors.count]

            // Box layer
            let box      = CAShapeLayer()
            box.path     = UIBezierPath(roundedRect: flipped, cornerRadius: 4).cgPath
            box.strokeColor = color.cgColor
            box.fillColor   = color.withAlphaComponent(0.08).cgColor
            box.lineWidth   = 2
            overlayView.layer.addSublayer(box)
            boxLayers.append(box)

            // Label layer
            let conf  = Int(top.confidence * 100)
            let text  = "\(top.identifier) \(conf)%"
            let label = CATextLayer()
            label.string           = text
            label.fontSize         = 11
            label.foregroundColor  = UIColor.white.cgColor
            label.backgroundColor  = color.cgColor
            label.cornerRadius     = 3
            label.contentsScale    = UIScreen.main.scale
            label.frame            = CGRect(x: flipped.minX, y: flipped.minY - 22,
                                            width: CGFloat(text.count * 7 + 10), height: 20)
            overlayView.layer.addSublayer(label)
            labelLayers.append(label)
        }

        let count = obs.count
        let info  = "\(count) object\(count == 1 ? "" : "s") · Objects365"
        hudLabel.text = fps.isEmpty ? "  \(info)  " : "  \(info) · \(fps)  "
    }
}

// ─────────────────────────────────────────────────────────
extension ObjectDetectionViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let req = detectionRequest,
              let px  = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: px, orientation: .right)
                .perform([req])
    }
}
